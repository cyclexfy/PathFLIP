"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torch.nn import LayerNorm
import numpy as np
from typing import Callable
import argparse

from .blip2 import Blip2Base
from .utils.dist_funs import pl_concat_all_gather, concat_all_gather, all_gather_with_grad
from .utils.utils import is_dist_avail_and_initialized

# from lavis.models.blip_models.blip_outputs import BlipOutput
# from .blip_outputs import BlipOutput
from .blip2 import Blip2Base

from .loss import create_loss


class pathflip_align(Blip2Base):
    def __init__(
        self,
        use_fg_loss=True,
        bert_name="/path/bert-base-uncased",
        text_max_length=512,
        temperature=0.1,
        path_enc = "Linear",
        num_query_token=32,
        num_hidden_layers=12,
        cross_attention_freq=2,
        path_input_dim=512,
        embed_dim=256,
        init_logit_scale=np.log(1/0.07),
        init_logit_bias=None,
        fg_use_proj=True,
        args=None,
    ):
        super().__init__()
        self.fg = use_fg_loss
        self.args = args
        self.temperature = temperature
        self.tokenizer = self.init_tokenizer(bert_name)
        self.path_encoder = self.init_path_encoder(
            input_dim=path_input_dim,
            emb_dim=embed_dim,
            model_name=path_enc
        )
        self.text_max_length = text_max_length
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token=num_query_token, 
            vision_width=embed_dim, 
            num_hidden_layers=num_hidden_layers, 
            cross_attention_freq=cross_attention_freq, 
            bert_name=bert_name
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        self.path_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # Transformer
        trans_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.path_trans = nn.TransformerEncoder(trans_layer, num_layers=2)

        # Fine-grained Loss
        if self.fg:
            self.fg_loss = create_loss(args)
            if fg_use_proj:
                self.fg_head = AttentionPoolingBlock(context_dim=embed_dim)
            else:
                self.fg_head = PureAttentionPoolingBlock(context_dim=embed_dim)
            self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
            if init_logit_bias is not None:
                self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
            else:
                self.logit_bias = None

    
    def forward(self, batch, return_attn=False):

        path = batch['path']
        text = batch['text']

        device = self.device
        batch_size = path.shape[0]

        if is_dist_avail_and_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        ### image
        path_local_embeds = self.path_encoder(path) # [B, L, S, D]
        b, l, s = path_local_embeds.shape[:-1]
        path_embeds_cluster = path_local_embeds.reshape(b, l*s, -1)
        path_mask = torch.ones(path_embeds_cluster.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=path_embeds_cluster,
            encoder_attention_mask=path_mask,
            # use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        path_global_feats = self.path_proj(torch.mean(query_output.last_hidden_state, dim=1)) # [B, D]
        global_attn = query_output.cross_attentions[-2] # global attn_scores

        path_feats_local_list = []
        local_attn = []
        for i in range(path_local_embeds.shape[1]):
            path_local_embeds_i = path_local_embeds[:, i, :, :]
            path_local_mask_i = torch.ones(path_local_embeds_i.shape[:-1]).to(device)
            query_output_local = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=path_local_embeds_i,
                encoder_attention_mask=path_local_mask_i,
                return_dict=True,
                output_attentions=True,
            )
            path_feats_local_list.append(query_output_local.last_hidden_state)
            local_attn.append(query_output_local.cross_attentions[-2])
        path_local_feats = torch.stack(path_feats_local_list, dim=1) # [B, L, num_q, D]
        path_local_feats = torch.mean(path_local_feats, dim=2)
        path_local_feats = self.path_proj(path_local_feats) # [B, L, D]
        local_attn = torch.stack(local_attn, dim=-2) # L * [B, num_head, num_query, num_patch] --> [B, num_head, num_query, L, num_patch]

        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]

        ### text
        text_tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.text_max_length, 
            return_tensors="pt"
        ).to(device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True
        )
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
    
        ### Image-Text Contrastive Loss ###
        loss_itc, sim_i2t, sim_t2i = self.contrast_global(
            path_feats=path_global_feats, 
            text_feats=text_feats, 
            rank=rank, 
            bs=batch_size, 
            device=device,
            need_norm=True
        )
        
        ### Fine-grained Loss ###
        loss_fg = 0.
        if self.fg:
            split_text = batch['split_text'] # [[first, second, ...], [..., ..., ...], ...]
            split_text_tokens_ids = []
            split_text_tokens_attn = []
            for text_list in split_text:
                tokens = self.tokenizer(
                    text_list,
                    padding='max_length',
                    truncation=True,
                    max_length=self.text_max_length//2,
                    return_tensors="pt"
                ).to(device)
                split_text_tokens_ids.append(tokens.input_ids)
                split_text_tokens_attn.append(tokens.attention_mask)

            split_text_tokens_ids = torch.stack(split_text_tokens_ids, dim=0) # [B, K, L]
            split_text_tokens_attn = torch.stack(split_text_tokens_attn, dim=0) # [B, K, L]
            b, k ,l = split_text_tokens_ids.shape
            split_text_tokens_ids = split_text_tokens_ids.reshape(b*k, l) # [B*K, L]
            split_text_tokens_attn = split_text_tokens_attn.reshape(b*k, l) # [B*K, L]
            split_text = batch['split_text']

            split_text_output = self.Qformer.bert(
                split_text_tokens_ids,
                attention_mask=split_text_tokens_attn,
                return_dict=True,
            )
            split_text_feats = self.text_proj(split_text_output.last_hidden_state[:, 0, :]) # [B*K, D]

            loss_fg = self.fg_loss(
                image_features = path_local_feats,
                text_features = split_text_feats,
                logit_scale=self.logit_scale,
                logit_bias=self.logit_bias,
                image_tokens=path_local_feats,
                visual_proj=self.fg_head,
                rank=rank,
                world_size=world_size,
                output_attn_weights=return_attn,
            )

            if return_attn:
                loss_fg, region_attn = loss_fg
 
        loss = loss_itc + loss_fg

        if return_attn:
            return loss, loss_itc, loss_fg, global_attn, local_attn, region_attn
        else:
            return loss, loss_itc, loss_fg

    ###============== Image-text Contrastive ===================###
    def contrast_global(self, path_feats, text_feats, rank, bs, device, need_norm=True):
        if need_norm:
            path_feats = F.normalize(path_feats, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)

        path_feats_all = concat_all_gather(path_feats)  # [batch_size*num_gpu, embed_dim]
        text_feat_all = concat_all_gather(text_feats)  # [batch_size*num_gpu, embed_dim]

        sim_i2t = torch.einsum("bd,nd->bn", path_feats, text_feat_all) # [batch_size, batch_size*num_gpu]
        sim_i2t = sim_i2t / self.temperature # [batch_size, batch_size*num_gpu]

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2i = torch.einsum("bd,nd->bn", text_feats, path_feats_all) # [batch_size, batch_size*num_gpu]
        sim_t2i = sim_t2i / self.temperature  # [batch_size, batch_size*num_gpu]

        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(device)
        targets = torch.arange(rank * bs, rank * bs + bs, dtype=torch.long, device=device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        return loss_itc, sim_i2t, sim_t2i


    def forward_image(self, batch, return_attn=False):

        path = batch['path']
        device = self.device
        batch_size = path.shape[0]

        ### image
        path_local_embeds = self.path_encoder(path) # [B, L, S, D]
        b, l, s = path_local_embeds.shape[:-1]
        path_embeds_cluster = path_local_embeds.reshape(b, l*s, -1)
        path_mask = torch.ones(path_embeds_cluster.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=path_embeds_cluster,
            encoder_attention_mask=path_mask,
            # use_cache=True,
            output_attentions=return_attn,
            return_dict=True,
        )
        path_global_feats = self.path_proj(torch.mean(query_output.last_hidden_state, dim=1)) # [B, D]
        if return_attn:
            global_attn = query_output.cross_attentions[-2] # global attn_scores

        path_feats_local_list = []
        local_attn = []
        for i in range(path_local_embeds.shape[1]):
            path_local_embeds_i = path_local_embeds[:, i, :, :]
            path_local_mask_i = torch.ones(path_local_embeds_i.shape[:-1]).to(device)
            query_output_local = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=path_local_embeds_i,
                encoder_attention_mask=path_local_mask_i,
                return_dict=True,
                output_attentions=return_attn,
            )
            path_feats_local_list.append(query_output_local.last_hidden_state)
            if return_attn:
                local_attn.append(query_output_local.cross_attentions[-2])
        path_local_feats = torch.stack(path_feats_local_list, dim=1) # [B, L, num_q, D]
        path_local_feats = torch.mean(path_local_feats, dim=2)
        path_local_feats = self.path_proj(path_local_feats) # [B, L, D]
        if return_attn:
            local_attn = torch.stack(local_attn, dim=-2) # L * [B, num_head, num_query, num_patch] --> [B, num_head, num_query, L, num_patch]

        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]

        if return_attn:
            return path_global_feats, global_attn, local_attn
        else:
            return path_global_feats

    
    def forward_text(self, batch):

        text = batch['text']
        device = self.device

        ### text
        text_tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.text_max_length, 
            return_tensors="pt"
        ).to(device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True
        )
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
    
        return text_feats


    def forward_attn(self, batch):

        path = batch['path']

        device = self.device
        batch_size = path.shape[0]

        ### image
        path_local_embeds = self.path_encoder(path) # [B, L, S, D]
        b, l, s = path_local_embeds.shape[:-1]
        path_embeds_cluster = path_local_embeds.reshape(b, l*s, -1)
        path_mask = torch.ones(path_embeds_cluster.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=path_embeds_cluster,
            encoder_attention_mask=path_mask,
            # use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        path_global_feats = self.path_proj(torch.mean(query_output.last_hidden_state, dim=1)) # [B, D]
        global_attn = query_output.cross_attentions[-2] # global attn_scores

        path_feats_local_list = []
        local_attn = []
        for i in range(path_local_embeds.shape[1]):
            path_local_embeds_i = path_local_embeds[:, i, :, :]
            path_local_mask_i = torch.ones(path_local_embeds_i.shape[:-1]).to(device)
            query_output_local = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=path_local_embeds_i,
                encoder_attention_mask=path_local_mask_i,
                return_dict=True,
                output_attentions=True,
            )
            path_feats_local_list.append(query_output_local.last_hidden_state)
            local_attn.append(query_output_local.cross_attentions[-2])
        path_local_feats = torch.stack(path_feats_local_list, dim=1) # [B, L, num_q, D]
        path_local_feats = torch.mean(path_local_feats, dim=2)
        path_local_feats = self.path_proj(path_local_feats) # [B, L, D]
        local_attn = torch.stack(local_attn, dim=-2) # L * [B, num_head, num_query, num_patch] --> [B, num_head, num_query, L, num_patch]

        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]

        ### text
        ### Fine-grained Attention ###
        if self.flair:
            split_text = batch['split_text'] # [[first, second, ...], [..., ..., ...], ...]
            split_text_tokens_ids = []
            split_text_tokens_attn = []
            for text_list in split_text:
                tokens = self.tokenizer(
                    text_list,
                    padding='max_length',
                    truncation=True,
                    max_length=self.text_max_length//2,
                    return_tensors="pt"
                ).to(device)
                split_text_tokens_ids.append(tokens.input_ids)
                split_text_tokens_attn.append(tokens.attention_mask)

            split_text_tokens_ids = torch.stack(split_text_tokens_ids, dim=0) # [B, K, L]
            split_text_tokens_attn = torch.stack(split_text_tokens_attn, dim=0) # [B, K, L]
            b, k ,l = split_text_tokens_ids.shape
            split_text_tokens_ids = split_text_tokens_ids.reshape(b*k, l) # [B*K, L]
            split_text_tokens_attn = split_text_tokens_attn.reshape(b*k, l) # [B*K, L]

            split_text_output = self.Qformer.bert(
                split_text_tokens_ids,
                attention_mask=split_text_tokens_attn,
                return_dict=True,
            )
            split_text_feats = self.text_proj(split_text_output.last_hidden_state[:, 0, :]) # [B*K, D]
            split_text_feats = split_text_feats.reshape(b, k, -1) # [B, K, D]

            local_image_features, region_attn= self.fg_head(
                q=split_text_feats, 
                k=path_local_feats, 
                v=path_local_feats, 
                output_attn_weights=True)
            
        return global_attn, local_attn, region_attn



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
        # self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True,
        #                                   add_zero_attn=True)
        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
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

class AttentionPoolingBlock(nn.Module):
    def __init__(
            self,
            context_dim: int,
            n_head: int = 8,
            norm_layer: Callable = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.w_q = nn.Linear(context_dim, context_dim)
        self.w_k = nn.Linear(context_dim, context_dim)
        self.w_v = nn.Linear(context_dim, context_dim) 
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

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=average_attn_weights)
            return out, attn_weights
        else:
            out = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
            return out
        

if __name__ == "__main__":
    from utils.process_args import get_args

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    from dataset.dataset_sw import sample_dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_args()
    args.use_flair_loss = True
    args.itm = False
    args.lm = False
    args.bert_name = "/path/biobert-base-cased-v1.2"

    text = ["The pathological findings reveal that the patient has an invasive high-grade urothelial carcinoma with infiltrating growth limited to the deep half of the muscularis propria of the bladder. The ureters and urethra are not involved, and no multicentricity within the bladder is identified. Vascular and perineural invasion are present. The surgical margins are clear of the tumor, and non-neoplastic mucosa shows ulceration and a foreign body reaction. Examination of the urethral margin shows benign urethral and prostate tissue. Both the right and left distal pelvic lymph nodes are benign, with 11 and 3 nodes examined respectively. The cystoprostatectomy specimen reveals an additional adenocarcinoma of the prostate graded as Gleason score 6 (3+3), involving multiple regions of the prostate including the capsule but not extending beyond it. Multicentric invasive carcinoma and high-grade prostatic intraepithelial neoplasia are also noted in the prostate. Neither the seminal vesicles nor the bladder neck are involved, and the surgical margins are free of tumor. The left and right distal ureters are benign. The final pathological staging classifies the bladder tumor as pT2b and the prostate adenocarcinoma as pT2b, both confined to their respective organs without extravesical or extraprostatic extension."]
    split_text = sample_dict(
        text=text,
        k=8,
        sampling_mode=args.sampling_mode,
        max_merged_num=args.max_merged_num,
        return_text=True
    )
    split_text = [split_text]
    path = torch.randn(1, 64, 256, 512).to(device)

    batch = {
        'path': path,
        'text': text,
        'split_text': split_text
    }

    model = pathflip_align(args=args).to(device)

    model_output = model(batch)
    loss, loss_itc, loss_fg = model_output
    print(f"loss_itc: {loss_itc}")
    print(f"loss_fg: {loss_fg}")

    # path_feats = model.forward_image(batch)
    # print(f"path_feats: {path_feats.shape}")

    # text_feats = model.forward_text(batch)
    # print(f"text_feats: {text_feats.shape}")

    # generated_text = model.generate_with_image(batch)
    # print(f"generated_text: {generated_text}")

    # global_attn, local_attn, region_attn = model.forward_attn(batch)
    # print(f"global_attn: {global_attn.shape}")
    # print(f"local_attn: {local_attn.shape}")
    # print(f"region_attn: {region_attn.shape}")