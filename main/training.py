# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right

# *user-defined
from models import gloss_free_model
from datasets import S2T_Dataset
import utils as utils

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER
from data import load_data
from model import build_model
def main (args, config): 
    print (args)
    device = torch.device(args.device)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    
    print(config)
    print(f"Creating dataset:")
    train_data, dev_data, test_data, txt_vocab, txt_field = load_data(config["data"], args)

    print (f"Creating model:")

    # Create model based on config
    model = build_model(
        cfg=config["model"],
        sgn_dim=512, 
        txt_vocab=txt_vocab,
        multimodal=False, 
        do_translation=True
    )

    # Create Optimizer and scheduler 


    # Create criterion 


    # Go through training epochs 

    






