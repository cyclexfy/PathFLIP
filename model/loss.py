import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

# try:
#     import horovod.torch as hvd
# except ImportError:
#     hvd = None


def create_loss(args):

    if args.use_fg_loss:
        return FGLoss(
            num_cap_per_img=args.text_sample_num,
            added_mps_loss=args.add_mps_loss, # True/False
        )
    else:
        raise NotImplementedError("Loss function for the given configuration is not implemented.")

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)



def get_multi_positive_mps(target, k):
    """
    :param target: tensor of shape (b, b*k), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b*k), for each row i, the col [i*k, (i+1)*k] should be ones
    """
    for i in range(target.shape[0]):
        target[i, i * k:(i + 1) * k] = 1
    return target



def get_multi_positive_tcs(target, k):
    """
    :param target: tensor of shape (b, b+k-1), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b+k-1), for each row i, the col [i, i+k) should be ones
    """
    for i in range(target.shape[0]):
        target[i, i: i + k] = 1
    return target

def get_mps_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, D)
    text_features: (B*K, D)
    """
    logits = logit_scale * image_features @ text_features.T  # if multi-cap: (B, B*K)
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_mps_ground_truth(device, dtype, target_shape, negative_only=False,
                                        num_captions=4):
    dim0, dim1 = target_shape  # (B, B*K)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B*K)
    if not negative_only:
        labels = get_multi_positive_mps(target=labels, k=num_captions)
    return labels

def get_intra_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, K, D),
    text_features: (B, K, D).
    Target: (B, K, K)
    """
    logits = logit_scale * torch.einsum('bkd,bjd->bkj', image_features, text_features)
    # logits = logit_scale * image_features @ text_features.T  
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_tcs_ground_truth(device, dtype, target_shape, negative_only=False, num_captions=4):
    dim0, dim1 = target_shape  # (B, B+K-1)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B+K-1)
    if not negative_only:
        labels = get_multi_positive_tcs(target=labels, k=num_captions)
    return labels

def get_tcs_logits(features_0, features_1, logit_scale, logit_bias=None):
    logits = logit_scale * torch.einsum('bij,bij->bi', features_0, features_1)
    if logit_bias is not None:
        logits += logit_bias
    return logits

### Fine-grained Loss
class FGLoss(nn.Module):
    def __init__(
            self,
            cache_labels=False,
            bidir=True,
            use_horovod=False,
            num_cap_per_img=8, # num_sampled_captions
            added_mps_loss=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        # self.rank = rank
        # self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}
        self.num_cap_per_img = num_cap_per_img

    def _loss_with_attn_pool(self, image_features, image_tokens, text_features, logit_scale,
                             logit_bias=None, negative_only=False, visual_proj=None, attn_mask=None, output_attn_weights=False):

        if output_attn_weights:
            local_image_features, attn_weights= visual_proj(text_features, image_tokens, image_tokens, attn_mask=attn_mask, output_attn_weights=output_attn_weights)  # (B, B+K-1, D)
        else:    
            local_image_features = visual_proj(text_features, image_tokens, image_tokens, attn_mask=attn_mask, output_attn_weights=output_attn_weights)  # (B, B+K-1, D)

        local_image_features = F.normalize(local_image_features, dim=-1)
        global_text_features = F.normalize(text_features, dim=-1)

        i2t_logits = get_tcs_logits(local_image_features, global_text_features, logit_scale, logit_bias)

        i2t_labels = get_tcs_ground_truth(device=text_features.device,
                                        dtype=text_features.dtype,
                                        target_shape=i2t_logits.size(),
                                        negative_only=negative_only,
                                        num_captions=self.num_cap_per_img)

        tcs_loss = -F.logsigmoid(i2t_labels * i2t_logits).sum() / text_features.shape[1] # text-conditioned sigmoid loss

        loss = tcs_loss

        if output_attn_weights:
            return loss, attn_weights
        else:
            return loss

    def forward(
            self,
            image_features, 
            text_features, 
            logit_scale, 
            logit_bias, 
            image_tokens,
            visual_proj=None, 
            output_dict=False, 
            attn_mask=None, 
            output_attn_weights=False,
            rank=0,
            world_size=1,
        ):
        '''
        expected shape: text_features: (B*K, D), image_embeddings: (B, L, D)
        '''
        if self.added_mps_loss:
            g_text_features = text_features  # (B*K, D)
        else:
            g_text_features = None
        

        # We don't change the shape of image tokens anywhere before the loss function.
        batch_size = image_tokens.shape[0]
        num_captions = self.num_cap_per_img
        caption_indices = torch.arange(batch_size * num_captions).view(batch_size, num_captions).to(
            text_features.device)
        
        text_features = downsample_text_features(text_features=text_features, batch_size=batch_size,
                                                 caption_indices=caption_indices,
                                                 num_captions=num_captions)

        loss_out = self._loss_with_attn_pool(image_features=image_features,
                                         image_tokens=image_tokens,
                                         text_features=text_features,
                                         visual_proj=visual_proj,
                                         logit_scale=logit_scale,
                                         logit_bias=logit_bias,
                                         attn_mask=attn_mask,
                                         output_attn_weights=output_attn_weights)
        if output_attn_weights:
            loss, attn_weights = loss_out
            attn_weights_out = [] 
            for i in range(text_features.shape[0]):
                attn_weights_out.append(attn_weights[i, i: i + self.num_cap_per_img])
            attn_weights_out = torch.stack(attn_weights_out, dim=0)
        else:
            loss = loss_out

        if world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (rank + 1) % world_size
            left_rank = (rank - 1 + world_size) % world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(world_size - 1, 2)

                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=f,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    
                    loss += self._loss_with_attn_pool(
                        image_features=image_features,
                        image_tokens=image_tokens,
                        text_features=text_features_recv,
                        visual_proj=visual_proj,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True)
            else:
                text_features_to_right = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features

                for i in range(world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss_with_attn_pool(
                        image_features=image_features,
                        image_tokens=image_tokens,
                        text_features=text_features_from_left,
                        visual_proj=visual_proj,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True)

                    text_features_to_right = text_features_from_left
                    
        loss = loss / world_size

        if output_attn_weights:
            return loss, attn_weights_out
        else:
            return {"contrastive_loss": loss} if output_dict else loss
    



def downsample_text_features(text_features, batch_size, caption_indices, num_captions):
    device = text_features.device
    own_caption_indices = caption_indices  # Shape: (B, K)

    mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)

    other_image_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(batch_size, batch_size)
    other_image_indices = other_image_indices[mask].view(batch_size, batch_size - 1)
    random_offsets = torch.randint(0, num_captions, (batch_size, batch_size - 1), device=device)  # (B, B-1)
    other_caption_indices = caption_indices[other_image_indices, random_offsets]  # sampled indices (B, B-1)

    combined_indices = torch.cat([own_caption_indices, other_caption_indices], dim=1) # (B, K+B-1)
    combined_indices, _ = combined_indices.sort(dim=1)
    flat_combined_indices = combined_indices.view(-1)  # flatten to take the text_features out

    # 确保 flat_combined_indices 在相同设备上，并且是整数类型
    flat_combined_indices = flat_combined_indices.to(device=text_features.device, dtype=torch.long)

    downsampled_text_features = text_features[flat_combined_indices] # (B*(K+B-1), D)

    embed_dim = text_features.shape[-1]  # Reshape to (B, K + B - 1, D)
    downsampled_text_features = downsampled_text_features.view(batch_size, num_captions + batch_size - 1, embed_dim)
    return downsampled_text_features


from typing import Callable
from torch.nn import LayerNorm

class PureAttentionPoolingBlock(nn.Module):
    """
    Just a pure attn_pooling implementation, without ln_post, without projection, no mormalized_final
    """

    def __init__(
            self,
            context_dim: int,
            n_head: int = 8,
            norm_layer: Callable = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True,
                                          add_zero_attn=True)
        # self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(context_dim)
        self.ln_k = norm_layer(context_dim)
        self.ln_v = norm_layer(context_dim)
        self.need_weights=need_weights
        self.n_head = n_head

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: torch.Tensor = None, output_attn_weights=False, average_attn_weights=True):
        batch_size, seg_length, embed_dim = k.size()
        _, query_length, _ = q.size()

        if attn_mask is not None:

            if attn_mask.size() != (batch_size, seg_length):
                expected_shape = (batch_size, seg_length)
                actual_shape = tuple(attn_mask.size())
                raise ValueError(f"Expected attn_mask shape to be {expected_shape}, but got {actual_shape}")
            
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.n_head, query_length, seg_length)
            attn_mask = attn_mask.reshape(batch_size * self.n_head, query_length, seg_length)
            attn_mask = (1.0 - attn_mask) * torch.finfo(attn_mask.dtype).min

        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=average_attn_weights)
            return out, attn_weights
        else:
            out = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
        # we can directly normalize the output, without setting a flag
        #return F.normalize(out, dim=-1)
            return out

