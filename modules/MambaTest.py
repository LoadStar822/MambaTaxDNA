#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 12:53
# @Author  : Qinzhong Tian
# @File    : MambaTax.py
# @Software: PyCharm
import os
import time
import numpy as np
import torch
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from torch import nn
from torch.utils.data import DataLoader
from modules.mamba_simple import Mamba
from utils.Tokenizer import DNATokenizer
from memory_profiler import profile
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from utils.tax_entry import TaxidLineage

class CustomGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, seq_len, superkingdom_to_label, phylum_to_label ,cache_lineage=True):
        self.phylum_to_label = phylum_to_label
        self.superkingdom_to_label = superkingdom_to_label
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.taxidLineage = TaxidLineage()

        if cache_lineage:
            taxids = [taxid for _, taxid in dataset]
            self.taxidLineage.populate(taxids, ['superkingdom', 'phylum'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq, taxid = self.dataset[idx]
        encoded_seq = self.tokenizer.encode(seq)
        encoded_seq = encoded_seq[:self.seq_len] + [self.tokenizer.pad_token_id] * (self.seq_len - len(encoded_seq))

        phylum = self.taxidLineage.get_ranks(taxid)['phylum'][1]
        superkingdom = self.taxidLineage.get_ranks(taxid)['superkingdom'][1]
        superkingdom_label = self.superkingdom_to_label[superkingdom]
        phylum_label = self.phylum_to_label[phylum]

        return (torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(superkingdom_label, dtype=torch.long),
                torch.tensor(phylum_label, dtype=torch.long))
class MambaTest(L.LightningModule):
    def __init__(self, train_dataset, valid_dataset, n_layer, d_model, seq_len,
                 lr, lr_scheduler_factor, weight_decay,
                 batch_size_train, batch_size_val, gpu_cnt):
        super().__init__()
        self.training_entries = train_dataset
        self.validation_entries = valid_dataset

        self.tokenizer = DNATokenizer()

        mamba_config = MambaConfig(n_layer=n_layer, d_model=d_model,
                                   vocab_size=self.tokenizer.vocab_size(),
                                   ssm_cfg={}, rms_norm=True, residual_in_fp32=True,
                                   fused_add_norm=True, pad_vocab_size_multiple=1)
        self.mambaDNA = MambaLMHeadModel(mamba_config)

        self.taxidLineage = TaxidLineage()

        self.superkingdom_to_label, self.label_to_superkingdom, self.num_superkingdoms = self.create_label_mapping('superkingdom')
        self.phylum_to_label, self.label_to_phylum, self.num_phylums = self.create_label_mapping('phylum')

        self.superkingdom_head = torch.nn.Linear(12, self.num_superkingdoms)
        self.phylum_head = torch.nn.Linear(12, self.num_phylums)

        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len = seq_len
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.weight_decay = weight_decay
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_superkingdoms)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_superkingdoms)

        self.save_hyperparameters(ignore=['mambaDNA'])

    def create_label_mapping(self, rank):
        combined_entries = self.training_entries + self.validation_entries
        unique_ranks = sorted(set(self.taxidLineage.get_ranks(taxid)[rank][1] for _, taxid in combined_entries))
        rank_to_label = {r: i for i, r in enumerate(unique_ranks)}
        label_to_rank = {i: r for r, i in rank_to_label.items()}
        return rank_to_label, label_to_rank, len(unique_ranks)

    def forward(self, inpts):
        outputs = self.mambaDNA(inpts)
        logits = outputs.logits[:, -1, :]
        logits_superkingdom = self.superkingdom_head(logits)
        logits_phylum = self.phylum_head(logits)
        return logits_superkingdom, logits_phylum

    def create_dataloader(self, dataset, batch_size, shuffle):
        dataset = CustomGenomeDataset(dataset, self.tokenizer, self.seq_len,
                                      self.superkingdom_to_label ,self.phylum_to_label)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          pin_memory=True, persistent_workers=False)

    def train_dataloader(self):
        return self.create_dataloader(self.training_entries, self.batch_size_train, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.validation_entries, self.batch_size_val, shuffle=False)

    def training_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast():
            inpts, trgts_superkingdom, trgts_phylum = batch  # 现在批次中包含两个标签
            logits_superkingdom, logits_phylum = self(inpts)  # 模型返回两个层次的logits

            # 分别计算两个层次的损失
            loss_superkingdom = self.loss_fn(logits_superkingdom, trgts_superkingdom)
            loss_phylum = self.loss_fn(logits_phylum, trgts_phylum)

            # 组合两个损失
            total_loss = loss_superkingdom + loss_phylum  # 可以调整权重

            # 日志记录
            self.log("train_loss", total_loss, on_step=False, on_epoch=True)

        # 计算并记录界的准确率
        preds_superkingdom = torch.argmax(logits_superkingdom, dim=1)
        correct_superkingdom = (preds_superkingdom == trgts_superkingdom).float().sum()
        acc_superkingdom = correct_superkingdom / trgts_superkingdom.size(0)
        self.log("train_acc_superkingdom", acc_superkingdom, on_step=False, on_epoch=True)

        # 计算并记录门的准确率
        preds_phylum = torch.argmax(logits_phylum, dim=1)
        correct_phylum = (preds_phylum == trgts_phylum).float().sum()
        acc_phylum = correct_phylum / trgts_phylum.size(0)
        self.log("train_acc_phylum", acc_phylum, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        # 获取记录的平均损失
        epoch_loss = self.trainer.logged_metrics.get("train_loss", 0)

        # 获取记录的界准确率
        epoch_acc_superkingdom = self.trainer.logged_metrics.get("train_acc_superkingdom", 0)

        # 获取记录的门准确率
        epoch_acc_phylum = self.trainer.logged_metrics.get("train_acc_phylum", 0)

        # 记录这些值（如果需要）
        self.log("train_loss_epoch", epoch_loss, on_step=False, on_epoch=True)
        self.log("train_acc_superkingdom_epoch", epoch_acc_superkingdom, on_step=False, on_epoch=True)
        self.log("train_acc_phylum_epoch", epoch_acc_phylum, on_step=False, on_epoch=True)
        # 打印日志
        print(
            f"\nEpoch finished - Avg loss: {epoch_loss:.4f}, Avg superkingdom accuracy: {epoch_acc_superkingdom:.4f}, Avg phylum accuracy: {epoch_acc_phylum:.4f}")

    def validation_step(self, batch, batch_idx):
        inpts, trgts_superkingdom, trgts_phylum = batch
        logits_superkingdom, logits_phylum = self(inpts)

        # 计算两个层次的损失
        loss_superkingdom = self.loss_fn(logits_superkingdom, trgts_superkingdom)
        loss_phylum = self.loss_fn(logits_phylum, trgts_phylum)

        # 组合两个损失
        total_loss = loss_superkingdom + loss_phylum  # 您可以调整权重

        # 计算准确率
        preds_superkingdom = torch.argmax(logits_superkingdom, dim=1)
        acc_superkingdom = (preds_superkingdom == trgts_superkingdom).float().mean()

        preds_phylum = torch.argmax(logits_phylum, dim=1)
        acc_phylum = (preds_phylum == trgts_phylum).float().mean()

        # 记录损失和准确率
        self.log('val_loss', total_loss, on_step=False, on_epoch=True)
        self.log('val_acc_superkingdom', acc_superkingdom, on_step=False, on_epoch=True)
        self.log('val_acc_phylum', acc_phylum, on_step=False, on_epoch=True)

        return {"val_loss": total_loss, "val_acc_superkingdom": acc_superkingdom, "val_acc_phylum": acc_phylum}

    def on_validation_epoch_end(self):
        # 聚合所有批次的损失和准确率
        avg_loss = self.trainer.logged_metrics.get("val_loss", torch.tensor(0.0))
        avg_acc_superkingdom = self.trainer.logged_metrics.get("val_acc_superkingdom", torch.tensor(0.0))
        avg_acc_phylum = self.trainer.logged_metrics.get("val_acc_phylum", torch.tensor(0.0))

        # 打印日志
        print(
            f"\nValidation - Avg loss: {avg_loss:.4f}, Avg superkingdom accuracy: {avg_acc_superkingdom:.4f}, Avg phylum accuracy: {avg_acc_phylum:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.mambaDNA.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                  factor=self.lr_scheduler_factor)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_acc_superkingdom'}
class FastPerplexity(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, loss):
        self.total_log_probs += loss.double()
        self.count += 1

    def compute(self):
        return torch.exp(self.total_log_probs / self.count.double())