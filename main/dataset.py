import os
import cv2
import numpy as np
import torch.utils.data.dataset as Dataset
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import random
from vidaug import augmentors as va

# Add these special token definitions
SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    if is_train:
        left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
    else:
        left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

    return (left, top, left + crop_size, top + crop_size), resize


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(Dataset.Dataset):
    def __init__(self, path, config, args, phase, training_refurbish=False):
        super().__init__()
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish
        self.raw_data = load_dataset_file(path)
        self.img_path = config['data']['img_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        self.list = [key for key, value in self.raw_data.items()]
        
        # Setup text tokenization
        self.txt_field = None  # Will be set after vocabulary is built
        
        # Video augmentation setup
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            sometimes(va.RandomTranslate(x=10, y=10)),
        ])

    def set_txt_field(self, txt_field):
        """Set the text field after vocabulary is built"""
        self.txt_field = txt_field

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        tgt_sample = sample['text']
        name_sample = sample['name']
        img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']])
        
        return name_sample, img_sample, tgt_sample
    

    def load_imgs(self, paths):

        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        if len(paths) > self.max_length:
            tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            new_paths = []
            for i in tmp:
                actl_path = paths[i]
                if self.config.training.tokens:
                    actl_path = os.path.splitext(actl_path)[0] + ".pth"
                new_paths.append(actl_path)
            paths = new_paths

        
        imgs = torch.zeros(len(paths),3, self.args.input_size,self.args.input_size)
        crop_rect, resize = data_augmentation(resize=(self.args.resize, self.args.resize), crop_size=self.args.input_size, is_train=(self.phase=='train'))

        batch_image = []
        for i,img_path in enumerate(paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)

        if self.phase == 'train':
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]
        
        return imgs
    
    def collate_fn(self, batch):
        """Custom collate function to handle batching of both images and text"""
        name_batch, img_tmp, tgt_batch = [], [], []
        
        for name_sample, img_sample, tgt_sample in batch:
            name_batch.append(name_sample)
            img_tmp.append(img_sample)
            tgt_batch.append(tgt_sample)

        # Process images
        max_len = max([len(vid) for vid in img_tmp])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad

        # Pad videos
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
            ),
            dim=0)
            for vid in img_tmp]
        
        img_tmp = [padded_video[i][0:video_length[i],:,:,:] for i in range(len(padded_video))]
        src_length_batch = torch.tensor([len(vid) for vid in img_tmp])
        img_batch = torch.cat(img_tmp, 0)

        # Create attention mask for images
        mask_gen = []
        for i in src_length_batch:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen,
                                 padding_value=self.txt_field.vocab.stoi[PAD_TOKEN], 
                                 batch_first=True)
        img_padding_mask = (mask_gen != self.txt_field.vocab.stoi[PAD_TOKEN]).long()

        #print("mask_gen", mask_gen)
        #print("img_padding_mask", img_padding_mask)

        # Process text
        if self.txt_field is not None:
            #print("Original texts:", tgt_batch)
            
            # Explicitly tokenize and process
            tokenized_texts = [text.split() for text in tgt_batch]  # Split into words
            #print("After tokenization:", tokenized_texts)
            
            txt_input = [self.txt_field.process([tokens]) for tokens in tokenized_texts]  # Process pre-tokenized text
            #print("After processing:", txt_input)
            
            txt_input = pad_sequence([t.squeeze(0) for t in txt_input], 
                                   batch_first=True, 
                                   padding_value=self.txt_field.vocab.stoi[PAD_TOKEN])
            
            # Create text mask
            txt_mask = (txt_input != self.txt_field.vocab.stoi[PAD_TOKEN]).long()
            
            # Prepare decoder input by shifting right
            decoder_input = self.shift_tokens_right(txt_input, 
                                                    self.txt_field.vocab.stoi[PAD_TOKEN],
                                                    self.txt_field.vocab.stoi[BOS_TOKEN])
        else:
            txt_input = None
            txt_mask = None
        print("txt_mask", txt_mask)
        print("txt_input", txt_input)
        return {
            'video': img_batch,
            'attention_mask': img_padding_mask,
            'name_batch': name_batch,
            'src_length': src_length_batch,
            'txt_input': txt_input,
            'txt_mask': txt_mask,
            'labels': txt_input
        }
    

    def shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """Shift input ids one token to the right, and wrap the last non-pad token (usually <eos>)."""
        print("pad_token_id", pad_token_id)
        print("decoder_start_token_id", decoder_start_token_id  )
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids



