import argparse


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', type=str, default="pathvl_qformer")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--itm', action='store_false', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_false', help='use language modeling or not', default=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--enriched_descrption', action='store_true', default=False)
    # devices
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    # parser.add_argument('--devices', type=str, default='1')

    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    parser.add_argument('--log_every_n_steps', type=int, default=50) 

    # data
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8) # 8
    parser.add_argument('--text_max_len', type=int, default=384) # 512
    parser.add_argument('--max_dataset_length', type=int, default=None)

    parser.add_argument('--data_path', type=str, default='/path/slide_window_data/')
    # default
    parser.add_argument('--path_sample_num', type=int, default=4096)
    # sldie window
    parser.add_argument('--slide_window_size', type=int, default=256)
    parser.add_argument('--path_sample_windows_num', type=int, default=24) # 256

    # Fine-grained Loss
    # parser.add_argument('--num_sampled_captions', type=int, default=8)
    parser.add_argument('--use_fg_loss', action='store_false', help='use fg loss or not', default=True)
    parser.add_argument('--add_mps_loss', action='store_true', help='use mps loss or not', default=False)

    parser.add_argument('--text_sample_num', type=int, default=8) # 8
    parser.add_argument('--sampling_mode', type=str, default='diverse_sampling')
    parser.add_argument('--max_merged_num', type=int, default=3)

    # train mode
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    # evaluation
    parser.add_argument('--rerank_cand_num', type=int, default=64)
    # model paramenters
    parser.add_argument('--path_input_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=256) # 256
    
    # Bert
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    parser.add_argument('--bert_name', type=str, default="/path/biobert-base-cased-v1.2")
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--cross_attention_freq', type=int, default=2) # 2
    parser.add_argument('--num_query_token', type=int, default=8) # 32
    parser.add_argument('--num_hidden_layers', type=int, default=12) # 12
    
    # llm
    parser.add_argument('--llm_model', type=str, default="/path/Qwen3-0.6B")

    # optimization
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_steps', type=int, default=200, help='optimizer warmup steps')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
    parser.add_argument('--init_checkpoint', type=str, default='')
    parser.add_argument('--retrieval_eval_epoch', type=int, default=10)

    parser.add_argument('--stage1_path', type=str, default='')
    parser.add_argument('--stage2_path', type=str, default='')

    parser.add_argument('--model_type', type=str, default='')

    args = parser.parse_args()
    return args
