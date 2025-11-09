import torch
import numpy as np
import json
import os
import re
import random

from typing import Dict, Sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from mmengine.dataset import DefaultSampler
from torch.utils.data.distributed import DistributedSampler
from types import SimpleNamespace


# split text caption
def split_caption(text):
    """Split captions by sentence-ending markers."""
    return [cap.strip() for cap in re.split(r'\n|</s>|[.]', text) if cap.strip()]

def random_sample_from_list(captions_list, k, merged_num=1):
    n = len(captions_list)
    if merged_num == 1:
        if n >= k:
            return random.sample(captions_list, k)
        else:  #minimizing caption dupilications
            return random.choices(captions_list, k=k)
            #return captions_list + random.sample(captions_list, k - n)
    elif merged_num >= n:
        return ['. '.join(captions_list)]
    else:
        sampled_list = []
        sampled_indices = draw_numbers(n=n - merged_num, k=k)
        for sampled_index in sampled_indices:
            sampled_list.append('. '.join(captions_list[sampled_index:sampled_index + merged_num]))
        return sampled_list

def draw_numbers(n, k=4):
    population = list(range(0, n))
    if n >= k:
        return random.sample(population, k)
    else:
        return random.choices(population, k=k)

def sample_dict(text, k=3, tokenizer=None, sampling_mode='diverse_sampling', pixelprose=True, max_merged_num=3, return_text=False):

    if sampling_mode == 'diverse_sampling':
        if pixelprose:
            # raw_caption = text["caption"]
            # captions_list = split_caption(raw_caption)
            if isinstance(text, list):
                text = text[0]
            captions_list = split_caption(text)
        else:
            captions_list = (text['raw_caption'] + text['shortIB_captions'] + text['longIB_captions'] +
                             text['shortSV_captions'] + text['longSV_captions'] +
                             text['shortLLA_captions'] + text['longLLA_captions'])
        n_captions = len(captions_list)
        sampled_sentences = []
        for _ in range(k):
            merged_num = random.randint(1, max_merged_num)
            if merged_num == 1:
                # Sample one caption
                sampled_sentence = random.choice(captions_list)
                sampled_sentences.append(sampled_sentence)
            else:
                prob_flag = 0.5 # 50% merging subsequent captions, 50% merging captions from random positions
                if random.random() < prob_flag:
                    sampled_sentence_list = random_sample_from_list(
                        captions_list, k=1, merged_num=merged_num)
                    sampled_sentences.extend(sampled_sentence_list)
                else:
                    # Randomly select captions to merge
                    if n_captions >= merged_num:
                        captions_to_merge = random.sample(captions_list, merged_num)
                    else:
                        captions_to_merge = [random.choice(captions_list) for _ in range(merged_num)]
                    # Merge the captions
                    sampled_sentence = '. '.join(captions_to_merge)
                    sampled_sentences.append(sampled_sentence)

        # tokenized_sentences = tokenizer(sampled_sentences)
        if return_text:
            return sampled_sentences

        tokenized_sentences = tokenizer(sampled_sentences, 
                                        padding='max_length', 
                                        max_length=256, 
                                        truncation=True, 
                                        return_tensors='pt')
        
        return tokenized_sentences
    else:
        raise NotImplementedError('Please select a valid sampling method')

def default_collate_fn(instances: Sequence[Dict]):
    # check instances
    if not instances:
        raise ValueError("Input instances list is empty")

    text = []
    split_text = []
    path_features = []
    path_mask = []

    for instance in instances:
        text.append(instance['text'])
        split_text.append(instance['split_text'])
        path_features.append(instance['path'])
        path_mask.append(instance['path_mask'])
    
    path_features = torch.stack(path_features)
    path_mask = torch.stack(path_mask)

    data_dict = {
        'text': text,
        'split_text': split_text,
        'path': path_features,
        'path_mask': path_mask
    }

    return data_dict

# Todo
def collate_fn_SW(instances: Sequence[Dict]):
    # check instances
    if not instances:
        raise ValueError("Input instances list is empty")

    text = []
    split_text = []
    path_features = []
    path_mask = []

    for instance in instances:
        text.append(instance['text'])
        split_text.append(instance['split_text'])
        path_features.append(instance['path'])
        path_mask.append(instance['path_mask'])
    
    path_features = torch.stack(path_features)
    path_mask = torch.stack(path_mask)

    data_dict = {
        'text': text,
        'split_text': split_text,
        'path': path_features,
        'path_mask': path_mask
    }

    return data_dict


class PathFLIP_Dataset_SW(Dataset):

    def __init__(self,
        data_path=None,
        path_sample=False,
        slide_window_size=256,
        path_sample_windows_num=256, # 256 x 256 = 65536
        max_dataset_length=None,
        tokenizer=None,
        text_sample_num=3,
        sampling_mode='diverse_sampling',
        max_merged_num=3,
        ):
        super().__init__()
    
        self.path_sample = path_sample
        self.slide_window_size = slide_window_size
        self.path_sample_windows_num = path_sample_windows_num
        # self.path_sample_num = path_sample_num
        self.tokenizer = tokenizer
        self.sampling_mode = sampling_mode
        self.max_merged_num = max_merged_num
        self.text_sample_num = text_sample_num

        if data_path.endswith('.json'):
            json_data = json.load(open(data_path))
        else:
            raise NotImplementedError

        # check json data
        valid_json_data = []
        for item in json_data:
            if 'image' in item and item['image']:
                feature_path = item['image'][0]
                if os.path.exists(feature_path):
                    valid_json_data.append(item)
        json_data = valid_json_data

        if max_dataset_length is not None and len(json_data)>max_dataset_length:
            json_data = json_data[:max_dataset_length]
        
        self.json_data = json_data

        
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_dict = self.json_data[index]
        data_return = {'text': None}

        # check conversations
        if (isinstance(data_dict.get('conversations'), list) and len(data_dict['conversations']) > 1) and data_dict['conversations'][1].get('from') == 'gpt':
            data_dict['text'] = data_dict['conversations'][1]['value']
        else:
            raise NotImplementedError
        
        if self.tokenizer is not None:

            data_return['text'] = self.tokenizer(data_dict['text'], 
                                                truncation=True, 
                                                padding='max_length', 
                                                max_length=512,
                                                return_tensors="pt")
            
            data_return['split_text'] = sample_dict(text=data_dict['text'],
                                                k=self.text_sample_num,
                                                tokenizer=self.tokenizer,
                                                sampling_mode=self.sampling_mode,
                                                max_merged_num=self.max_merged_num)
        else:
            data_return['text'] = data_dict['text']
            
            data_return['split_text'] = sample_dict(text=data_dict['text'],
                                                k=self.text_sample_num,
                                                sampling_mode=self.sampling_mode,
                                                max_merged_num=self.max_merged_num,
                                                return_text=True)

        if data_dict.get('image', None) is not None:
            image_list = data_dict['image']
        if isinstance(image_list, str):
            image_list = [image_list]

        patch_features = []
        # load wsi features
        for image_file in image_list:
            if image_file.endswith('.pt'):
                slide_feature = torch.load(image_file, weights_only=True)
                patch_features.append(slide_feature)

        if len(patch_features) != 0:
            patch_features = torch.cat(patch_features, dim=0)

        slide_window_num = patch_features.shape[0] // self.slide_window_size
        patch_features = patch_features[:slide_window_num * self.slide_window_size]

        if patch_features.numel() == 0:
            print(f"Error: File {image_list[0]} resulted in an empty patch_features tensor, cannot reshape.")

        patch_features = patch_features.reshape(slide_window_num, self.slide_window_size, -1)

        if self.path_sample:
            # random sample patches
            max_windows_num = self.path_sample_windows_num
            n_windows = min(patch_features.shape[0], max_windows_num)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_windows, replace=False))
            patch_features = patch_features[idx, :, :]
            if n_windows < max_windows_num:
                ### random-select-padding
                num_to_pad = max_windows_num - n_windows
                pad_indices = np.random.choice(patch_features.shape[0], num_to_pad)
                pad_features = patch_features[pad_indices]
                patch_features = torch.cat([patch_features, pad_features], dim=0)
        patch_mask = torch.ones(patch_features.shape[0])

        data_return['path'] = patch_features
        data_return['path_mask'] = patch_mask

        return data_return


class pathVL_Dataset_dm(LightningDataModule):
    def __init__(
        self,
        data_path=None,
        path_sample=True,
        slide_window_size=256,
        path_sample_windows_num=256,
        tokenizer=None,
        max_dataset_length=None,
        batch_size: int = 4,
        num_workers: int = 4,
        args=None,
    ):
        super().__init__()
        self.path_sample = path_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

        self.train_dataset = PathFLIP_Dataset_SW(
            data_path=os.path.join(data_path, 'train_data.json'),
            path_sample=path_sample,
            slide_window_size=slide_window_size,
            path_sample_windows_num=path_sample_windows_num,
            max_dataset_length=max_dataset_length,
            tokenizer=tokenizer,
            text_sample_num=args.text_sample_num,
            sampling_mode=args.sampling_mode,
            max_merged_num=args.max_merged_num,
        )

        self.val_dataset = PathFLIP_Dataset_SW(
            data_path=os.path.join(data_path, 'test_data.json'),
            path_sample=path_sample,
            slide_window_size=slide_window_size,
            path_sample_windows_num=path_sample_windows_num,
            max_dataset_length=max_dataset_length,
            tokenizer=tokenizer,
            text_sample_num=args.text_sample_num,
            sampling_mode=args.sampling_mode,
            max_merged_num=args.max_merged_num,
        )
        # self.test_dataset = None

        self.collate_fn = default_collate_fn
    
    def train_dataloader(self):

        # 判断是否是分布式训练
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.train_dataset)
            shuffle = False  # 分布式 sampler 内部 shuffle，不需要 DataLoader 再 shuffle
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )
        
        return train_loader

    def val_dataloader(self):

        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return val_loader


# Test
if __name__ == '__main__':
    from utils.process_args import get_args
    args = get_args()
    dm = pathVL_Dataset_dm(
        data_path=args.data_path,
        slide_window_size=args.slide_window_size,
        path_sample_windows_num=args.path_sample_windows_num,
        max_dataset_length=args.max_dataset_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args
    )
    data = dm.train_dataloader().dataset[0]
    print(data)