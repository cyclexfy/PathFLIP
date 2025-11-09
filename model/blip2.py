"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os

import torch
import torch.nn as nn


from transformers import BertTokenizer

from nystrom_attention import NystromAttention
import numpy as np

# from lavis.models.base_model import BaseModel
from .base_model import BaseModel
from .Qformer import BertConfig, BertLMHeadModel
# from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

    
class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, bert_name="bert-base-uncased"):
        # 设置 bert_name
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(
        cls, 
        num_query_token, 
        vision_width, 
        num_hidden_layers=12, 
        add_cross_attention=True,
        cross_attention_freq=2, 
        bert_name="bert-base-uncased"
    ):
        # 设置 bert_name
        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = num_hidden_layers
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = add_cross_attention
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(
            bert_name, config=encoder_config, device_map=None,
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        state_dict = Qformer.state_dict()
        for name, param in Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        return Qformer, query_tokens

    def init_path_encoder(self, input_dim, emb_dim, model_name='Linear'):
        if model_name == "Linear":
            path_encoder = nn.Linear(input_dim, emb_dim)
        elif model_name == "MLP":
            path_encoder = nn.Sequential(nn.Linear(input_dim, emb_dim),
                                         nn.LayerNorm(emb_dim),
                                         nn.ReLU(),
                                         nn.Linear(emb_dim, emb_dim),
                                         nn.LayerNorm(emb_dim))
        elif model_name == "TransMIL":
            path_encoder = TransMIL(input_dim=input_dim, emb_dim=emb_dim)
        else:
            raise ValueError(f"{model_name} is not supported.")
            
        return path_encoder


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim=1024, emb_dim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=emb_dim)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.layer1 = TransLayer(dim=emb_dim)
        self.layer2 = TransLayer(dim=emb_dim)
        self.norm = nn.LayerNorm(emb_dim)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]
        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        #---->cls_token
        h = self.norm(h)[:,0]

        # local_feature, global_feature
        return h[:, 0], h[:, 1:]

