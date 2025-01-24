# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# *user-defined

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

from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimization
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc='Training')
        for batch in progress_bar:
            # Move batch to device
            video = batch['video'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()

            # this part might need to change 
            outputs = self.model(
                video=video,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if wandb.run:
                wandb.log({'train_batch_loss': loss.item()})
                
        return total_loss / len(self.train_dataloader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Validating'):
                video = batch['video'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    video=video,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
        return total_loss / len(self.val_dataloader)
    
    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Log metrics
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if wandb.run:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch
                })
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                print("Early stopping triggered")
                break
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'model_epoch_{epoch}_loss_{val_loss:.4f}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

def main(args, config):
    # Set device
    device = torch.device(args.device)
    
    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load data
    train_data, dev_data, test_data, txt_vocab, txt_field = load_data(config["data"], args)
    
    # Build model
    model = build_model(
        cfg=config["model"],
        sgn_dim=512,
        txt_vocab=txt_vocab,
        multimodal=False,
        do_translation=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_data,
        val_dataloader=dev_data,
        config=config
    )
    
    # Train
    trainer.train()






