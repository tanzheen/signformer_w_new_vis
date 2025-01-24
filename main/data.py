import os
import sys
import random

import torch
from torchtext import data
from torch.utils.data.dataset import Dataset
from torchtext.data import Field
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
    txt_field = data.Field(
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
        dataset=train_data,
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


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter