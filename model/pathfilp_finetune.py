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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel


from .blip2 import Blip2Base
from .utils.dist_funs import pl_concat_all_gather, concat_all_gather, all_gather_with_grad
from .utils.utils import is_dist_avail_and_initialized

# from lavis.models.blip_models.blip_outputs import BlipOutput
# from .blip_outputs import BlipOutput
from .blip2 import Blip2Base

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_default_dtype(torch.bfloat16)


def get_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side='right')
    tokenizer.add_special_tokens({
        'pad_token': '[pad]',
        'bos_token': '[bos]',
        'unk_token': '[unk]',
        'additional_special_tokens': ['<image>']
    })
    return tokenizer

def get_prompt(prompt_type='generate_caption'):
    if prompt_type == 'generate_caption':
        prompt = "Instruction: Describe the input pathology whole slide image.\n" \
        "Input pathology whole slide image: <image>.\n" \
        "Response: "
        return prompt
    else:
        raise NotImplementedError()

class pathflip_finetune(Blip2Base):
    def __init__(
        self,
        bert_name="/path/bert-base-uncased",
        text_max_length=512,
        path_enc = "Linear",
        num_query_token=32,
        num_hidden_layers=12,
        cross_attention_freq=2,
        path_input_dim=512,
        embed_dim=256,
        llm_model="/path/Qwen3-0.6B",
        llm_tuning='lora',
        args=None,
    ):
        super().__init__()

        self.args = args

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

        ### llm model
        self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model,
                    torch_dtype="auto",
                    device_map=None
        )
        config = AutoConfig.from_pretrained(llm_model)
        llm_hidden_size = config.hidden_size # 1024

        self.llm_tokenizer = get_tokenizer(llm_model)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.llm_tokenizer.image_token_id = self.llm_tokenizer("<image>", add_special_tokens=False).input_ids[0]

        ### set llm_model tuning
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.05
        if llm_tuning == 'lora':
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            )
            self.peft_config = peft_config
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            self.llm_model.print_trainable_parameters()
        elif llm_tuning == 'full':
            pass
        elif llm_tuning == 'freeze':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError()

        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id
        self.generate_caption_prompt = get_prompt(prompt_type='generate_caption')

        self.path_proj_llm = nn.Linear(embed_dim, llm_hidden_size)


    
    def forward(self, batch, return_attn=False):

        path = batch['path']
        text = batch['text']

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
        local_attn = torch.stack(local_attn, dim=1) # [B, L, num_head, S]

        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]


        prompt_tokens = self.llm_tokenizer(
            self.generate_caption_prompt, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).to(device) # [1, L]
        # get index
        pos = (prompt_tokens.input_ids == self.llm_tokenizer.image_token_id).nonzero(as_tuple=True)[1]
        prompt_tokens.input_ids = prompt_tokens.input_ids.repeat(batch_size, 1)
        prompt_tokens.attention_mask = prompt_tokens.attention_mask.repeat(batch_size, 1)
        prompt_tokens_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)

        path_feats_llm = self.path_proj_llm(path_feats)
        inputs_embeds_1 = torch.cat([prompt_tokens_embeds[:, pos, :], path_feats_llm, prompt_tokens_embeds[:, pos+1:, :]], dim=1)
        attention_mask_1 = torch.cat([
            prompt_tokens.attention_mask[:, pos:pos+1],
            torch.ones(path_feats_llm.shape[:-1], dtype=torch.long, device=device), 
            prompt_tokens.attention_mask[:, pos+1:]
            ], dim=1)
        
        text_tokens = self.llm_tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt"
        ).to(device)

        inputs_embeds_2 = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        attention_mask_2 = text_tokens.attention_mask

        inputs_embeds = torch.cat([inputs_embeds_1, inputs_embeds_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)

        targets_1 = torch.full(inputs_embeds_1.shape[:-1], -100).to(device)
        targets_2 = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100)
        targets = torch.cat([targets_1, targets_2], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            use_cache=True,
            return_dict=True
        )
        loss = outputs.loss

        return loss

    def forward_image(self, batch):

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
            return_dict=True,
        )
        path_global_feats = self.path_proj(torch.mean(query_output.last_hidden_state, dim=1)) # [B, D]

        path_feats_local_list = []
        for i in range(path_local_embeds.shape[1]):
            path_local_embeds_i = path_local_embeds[:, i, :, :]
            path_local_mask_i = torch.ones(path_local_embeds_i.shape[:-1]).to(device)
            query_output_local = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=path_local_embeds_i,
                encoder_attention_mask=path_local_mask_i,
                return_dict=True,
            )
            path_feats_local_list.append(query_output_local.last_hidden_state)
        path_local_feats = torch.stack(path_feats_local_list, dim=1) # [B, L, num_q, D]
        path_local_feats = torch.mean(path_local_feats, dim=2)
        path_local_feats = self.path_proj(path_local_feats) # [B, L, D]

        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]

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

    @torch.no_grad()
    def generate(
        self,
        batch,
        do_sample=True,
        use_nucleus_sampling=False,
        num_beams=3,
        max_new_tokens=512,
        min_new_tokens=128,
        length_penalty=1.0,
        repetition_penalty=1.0,
        num_captions=3
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

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
            return_dict=True,
        )
        path_global_feats = self.path_proj(torch.mean(query_output.last_hidden_state, dim=1)) # [B, D]

        path_feats_local_list = []

        for i in range(path_local_embeds.shape[1]):
            path_local_embeds_i = path_local_embeds[:, i, :, :]
            path_local_mask_i = torch.ones(path_local_embeds_i.shape[:-1]).to(device)
            query_output_local = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=path_local_embeds_i,
                encoder_attention_mask=path_local_mask_i,
                return_dict=True,
            )
            path_feats_local_list.append(query_output_local.last_hidden_state)
            
        path_local_feats = torch.stack(path_feats_local_list, dim=1) # [B, L, num_q, D]
        path_local_feats = torch.mean(path_local_feats, dim=2)
        path_local_feats = self.path_proj(path_local_feats) # [B, L, D]
 
        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]

        prompt_tokens = self.llm_tokenizer(
            self.generate_caption_prompt, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).to(device) # [1, L]
        # get index
        pos = (prompt_tokens.input_ids == self.llm_tokenizer.image_token_id).nonzero(as_tuple=True)[1]
        prompt_tokens.input_ids = prompt_tokens.input_ids.repeat(batch_size, 1)
        prompt_tokens.attention_mask = prompt_tokens.attention_mask.repeat(batch_size, 1)
        prompt_tokens_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)

        path_feats_llm = self.path_proj_llm(path_feats)
        inputs_embeds = torch.cat([prompt_tokens_embeds[:, pos, :], path_feats_llm, prompt_tokens_embeds[:, pos+1:, :]], dim=1)
        attention_mask = torch.cat([
            prompt_tokens.attention_mask[:, pos:pos+1],
            torch.ones(path_feats_llm.shape[:-1], dtype=torch.long, device=device), 
            prompt_tokens.attention_mask[:, pos+1:]
            ], dim=1)
                
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        captions = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return captions

    @torch.no_grad()
    def generate_with_instruction(
        self,
        batch,
        do_sample=True,
        use_nucleus_sampling=False,
        num_beams=3,
        max_new_tokens=128,
        length_penalty=1.0,
        repetition_penalty=1.0,
        num_captions=3
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        path = batch['path']
        instruction = batch['instruction']

        device = self.device
        batch_size = path.shape[0]

        prompt = f"Input pathology whole slide image: <image>. \nInstruction: {instruction} \nResponse:"

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
            return_dict=True,
        )
        path_global_feats = self.path_proj(torch.mean(query_output.last_hidden_state, dim=1)) # [B, D]

        path_feats_local_list = []

        for i in range(path_local_embeds.shape[1]):
            path_local_embeds_i = path_local_embeds[:, i, :, :]
            path_local_mask_i = torch.ones(path_local_embeds_i.shape[:-1]).to(device)
            query_output_local = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=path_local_embeds_i,
                encoder_attention_mask=path_local_mask_i,
                return_dict=True,
            )
            path_feats_local_list.append(query_output_local.last_hidden_state)
            
        path_local_feats = torch.stack(path_feats_local_list, dim=1) # [B, L, num_q, D]
        path_local_feats = torch.mean(path_local_feats, dim=2)
        path_local_feats = self.path_proj(path_local_feats) # [B, L, D]
 
        path_feats = self.path_trans(torch.cat([path_global_feats.unsqueeze(1), path_local_feats], dim=1)) # [B, L+1, D]
        path_global_feats = path_feats[:, 0, :]
        path_local_feats = path_feats[:, 1:, :]

        prompt_tokens = self.llm_tokenizer(
            prompt, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).to(device) # [1, L]
        # get index
        pos = (prompt_tokens.input_ids == self.llm_tokenizer.image_token_id).nonzero(as_tuple=True)[1]
        prompt_tokens.input_ids = prompt_tokens.input_ids.repeat(batch_size, 1)
        prompt_tokens.attention_mask = prompt_tokens.attention_mask.repeat(batch_size, 1)
        prompt_tokens_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)

        path_feats_llm = self.path_proj_llm(path_feats)
        inputs_embeds = torch.cat([prompt_tokens_embeds[:, pos, :], path_feats_llm, prompt_tokens_embeds[:, pos+1:, :]], dim=1)
        attention_mask = torch.cat([
            prompt_tokens.attention_mask[:, pos:pos+1],
            torch.ones(path_feats_llm.shape[:-1], dtype=torch.long, device=device), 
            prompt_tokens.attention_mask[:, pos+1:]
            ], dim=1)
                
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        captions = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return captions



if __name__ == "__main__":
    from utils.process_args import get_args

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    from dataset.dataset_sw import sample_dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_args()

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

    model = pathflip_finetune(args=args).to(device)

    # model_output = model(batch)
    # loss = model_output
    # print(f"loss: {loss}")

    path_feats = model.forward_image(batch)
    print(f"path_feats: {path_feats.shape}")

    text_feats = model.forward_text(batch)
    print(f"text_feats: {text_feats.shape}")

    # generated_text = model.generate_with_image(batch)
    # print(f"generated_text: {generated_text}")