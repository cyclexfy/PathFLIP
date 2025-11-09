import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib

from utils.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from typing import Dict, Any
from .utils.help_funcs import AttrDict
from .pathflip_align import pathflip_align

class pl_pathflip_align(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        self.args = args
        # name
        self.pathflip_align = pathflip_align(
            use_flair_loss=args.use_flair_loss,
            bert_name=args.bert_name,
            text_max_length=args.text_max_len,
            temperature=args.temperature,
            num_query_token=args.num_query_token,
            cross_attention_freq=args.cross_attention_freq,
            num_hidden_layers=args.num_hidden_layers,
            path_input_dim=args.path_input_dim,
            embed_dim=args.embed_dim,
            args=args
        )
        self.save_hyperparameters(args)


    def forward(self, batch):
        loss, loss_itc, loss_flair = self.pathflip_align(batch)
        return loss
    
    def training_step(self, batch):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch['path'].size(0)

        loss, loss_itc, loss_flair = self.pathflip_align(batch)

        ##============== Overall Loss ===================###
        self.log("train_loss_itc", float(loss_itc), batch_size=batch_size, sync_dist=True, on_epoch=True, on_step=False)
        self.log("train_loss_flair", float(loss_flair), batch_size=batch_size, sync_dist=True, on_epoch=True, on_step=False)
        self.log("train_loss", float(loss), batch_size=batch_size, sync_dist=True, on_epoch=True, on_step=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        batch_size = batch['path'].size(0)
        
        loss, loss_itc, loss_flair = self.pathflip_align(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss_itc", float(loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_flair", float(loss_flair), batch_size=batch_size, sync_dist=True)
        self.log("val_loss", float(loss), batch_size=batch_size, sync_dist=True)

        return loss


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