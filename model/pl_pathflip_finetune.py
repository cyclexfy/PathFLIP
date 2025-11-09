import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib
import torch.distributed as dist
import os
import json

from utils.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from typing import Dict, Any
from .utils.help_funcs import AttrDict, caption_evaluate
from .pathfilp_finetune import pathflip_finetune

class pl_pathflip_finetune(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        self.args = args
        # name
        self.pathflip_finetune = pathflip_finetune(
            bert_name=args.bert_name,
            text_max_length=args.text_max_len,
            num_query_token=args.num_query_token,
            cross_attention_freq=args.cross_attention_freq,
            num_hidden_layers=args.num_hidden_layers,
            path_input_dim=args.path_input_dim,
            embed_dim=args.embed_dim,
            llm_model=args.llm_model,
            args=args
        )

        self.save_hyperparameters(args)
        self.test_step_outputs = []


    def forward(self, batch):
        loss = self.pathflip_finetune(batch)
        return loss
    
    def training_step(self, batch):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch['path'].size(0)

        loss = self.pathflip_finetune(batch)

        ##============== Overall Loss ===================###
        self.log("train_loss", float(loss), batch_size=batch_size, sync_dist=True, on_epoch=True, on_step=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        batch_size = batch['path'].size(0)
        
        loss = self.pathflip_finetune(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss", float(loss), batch_size=batch_size, sync_dist=True)

        return loss

    @torch.no_grad()
    def test_step(self, batch):
        ###============== Captioning Results ===================###
        predictions = self.pathflip_finetune.generate(
            batch,
            num_captions=1
        )
        self.test_step_outputs.append((predictions, batch['text']))
        return predictions, batch['text']

    # def on_test_epoch_end(self):
    #     outputs = self.test_step_outputs
    #     list_predictions, list_targets = zip(*outputs)
    #     predictions = [i for ii in list_predictions for i in ii]
    #     targets = [i for ii in list_targets for i in ii]

    #     all_predictions = [None for _ in range(self.trainer.world_size)]
    #     all_targets = [None for _ in range(self.trainer.world_size)]

    #     dist.all_gather_object(all_predictions, predictions)
    #     dist.all_gather_object(all_targets, targets)
    #     if self.global_rank == 0:
    #         all_predictions = [i for ii in all_predictions for i in ii]
    #         all_targets = [i for ii in all_targets for i in ii]
    #         self.save_predictions(all_predictions, all_targets)
    #         ## fixme: I am not sure if the max length is the same as previous experiments
    #         bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
    #             caption_evaluate(all_predictions, all_targets, self.tokenizer, self.args.max_new_tokens)
    #         self.log("bleu2", bleu2, sync_dist=False)
    #         self.log("bleu4", bleu4, sync_dist=False)
    #         self.log("rouge_1", rouge_1, sync_dist=False)
    #         self.log("rouge_2", rouge_2, sync_dist=False)
    #         self.log("rouge_l", rouge_l, sync_dist=False)
    #         self.log("meteor_score", meteor_score, sync_dist=False)
    
    # def save_predictions(self, predictions, targets):
    #     assert len(predictions) == len(targets)
    #     with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
    #         for p, t in zip(predictions, targets):
    #             line = {'prediction': p, 'target': t}
    #             f.write(json.dumps(line, ensure_ascii=True) + '\n')


    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def load_from_stage1_checkpoint(self, path):
        print('loading from stage1 checkpoint')
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('pathflip_align.')[1]: v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.pathflip_finetune.load_state_dict(state_dict, strict=False)
        print('missing keys')
        print(missing_keys)
        print('unexpected keys')
        print(unexpected_keys)
        return self