import os
import sys
import random

import torch


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import socket
from dataset import SignTranslationDataset
from vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)

class Field:
    """Custom Field class for text processing and numericalization"""
    def __init__(
        self,
        init_token=None,
        eos_token=None,
        pad_token=None,
        unk_token=None,
        tokenize=None,
        batch_first=False,
        lower=False,
        include_lengths=False,
    ):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.tokenize = tokenize if tokenize is not None else str.split
        self.batch_first = batch_first
        self.lower = lower
        self.include_lengths = include_lengths
        self.vocab = None

    def preprocess(self, x):
        """Preprocess text by lowercasing and tokenizing"""
        if isinstance(x, list):  # Handle pre-tokenized input
            return x if not self.lower else [t.lower() for t in x]
        if self.lower:
            x = x.lower()
        return self.tokenize(x)

    def process(self, batch, device=None):
        """Convert batch of text to tensor of indices"""
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, batch):
        """Pad batch of sequences to same length"""
        max_len = max(len(x) for x in batch)
        
        # Add special tokens if specified
        pad_token = self.pad_token
        init_token = self.init_token
        eos_token = self.eos_token
        
        padded = []
        lengths = []
        
        for x in batch:
            tokens = []
            if init_token is not None:
                tokens.append(init_token)
            tokens.extend(x)
            if eos_token is not None:
                tokens.append(eos_token)
                
            length = len(tokens)
            tokens.extend([pad_token] * (max_len - length + 2))  # +2 for possible init/eos tokens
            padded.append(tokens)
            lengths.append(length)
            
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Convert text tokens to indices"""
        if self.vocab is None:
            raise RuntimeError("Vocab not set for Field")
            
        # Ensure each token is a string before lookup
        nums = [[self.vocab.stoi.get(str(x) if not isinstance(x, str) else x, 
                self.vocab.stoi[self.unk_token]) for x in ex] for ex in arr]
        var = torch.tensor(nums, device=device)
        
        if self.include_lengths:
            lengths = torch.tensor([len(x) for x in arr], device=device)
            return var, lengths
        return var

def load_data(data_cfg: dict, args) -> (Dataset, Dataset, Dataset, Vocabulary, Field):
    """
    Load train, dev and test data as specified in configuration.
    """
    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    
    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()
    
    # Create text field with tokenization and special tokens
    txt_field = Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    # Load datasets
    train_data = SignTranslationDataset(
        data_cfg['train_path'], data_cfg, args, phase='train'
    )
    train_data.set_txt_field(txt_field)
    # Build vocabulary from training data
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)
    txt_vocab_file = data_cfg.get("txt_vocab", None)
    
    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data.raw_data,
        vocab_file=txt_vocab_file,
    )
    
    # Assign vocabulary to text field
    txt_field.vocab = txt_vocab
    
    # Load dev and test data
    dev_data = SignTranslationDataset(
        data_cfg['dev_path'], data_cfg, args, phase='dev'
    )
    test_data = SignTranslationDataset(
        data_cfg['test_path'], data_cfg, args, phase='test'
    )
    dev_data.set_txt_field(txt_field)
    test_data.set_txt_field(txt_field)

    return train_data, dev_data, test_data, txt_vocab, txt_field


def token_batch_size_fn(new, count):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)




