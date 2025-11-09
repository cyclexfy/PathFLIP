import torch
import warnings
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

from pytorch_lightning import Trainer, strategies
from pytorch_lightning.loggers import CSVLogger
from datetime import datetime

from model.pl_pathflip_finetune import pl_pathflip_finetune

from dataset.dataset_sw import PathFLIP_Dataset_SW
from utils.process_args import get_args
from model.utils.dist_funs import MyDeepSpeedStrategy

## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

logging.getLogger("torch").setLevel(logging.ERROR)

def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = pl_pathflip_finetune.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = pl_pathflip_finetune(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = pl_pathflip_finetune(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    datamodule = PathFLIP_Dataset_SW(
        data_path=args.data_path,
        slide_window_size=args.slide_window_size,
        path_sample_windows_num=args.path_sample_windows_num,
        max_dataset_length=args.max_dataset_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args
    )

    callbacks = []
    timestamp = datetime.now().strftime("%m%d_%H%M")
    callbacks.append(plc.ModelCheckpoint(
        dirpath=f"all_checkpoints/{args.filename}_{timestamp}",
        filename='{epoch:02d}', 
        every_n_epochs=args.save_every_n_epochs, 
        save_top_k=-1,
        save_on_train_epoch_end=True))
    
    if len(args.devices.split(','))>1 or int(args.devices) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
    
    logger = CSVLogger(save_dir=f"all_checkpoints/{args.filename}_{timestamp}")
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_val_batches=1.0,
        log_every_n_steps=args.log_every_n_steps
    )
    
    if args.mode in ['train']:
        trainer.fit(model, datamodule=datamodule)
        trainer.validate(model, datamodule=datamodule)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.max_epochs - 1
        trainer.validate(model, datamodule=datamodule)
    else:
        raise NotImplementedError()

    return

if __name__ == '__main__':
    args = get_args()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)