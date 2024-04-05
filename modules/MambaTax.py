#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 12:53
# @Author  : Qinzhong Tian
# @File    : MambaTax.py
# @Software: PyCharm
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

from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from utils.tax_entry import TaxidLineage



class CustomGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, seq_len, class_weights=None, cache_lineage=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.taxidLineage = TaxidLineage()
        self.weight_classes = class_weights
        self.seq, self.taxid, self.superkingdom = zip(*dataset)

        # 为 superkingdom 创建标签映射
        unique_superkingdoms = sorted(set(self.superkingdom))
        self.superkingdom_to_label = {sk: i for i, sk in enumerate(unique_superkingdoms)}
        self.label_to_superkingdom = {i: sk for sk, i in self.superkingdom_to_label.items()}

        # 创建taxid到连续整数的映射
        unique_taxids = sorted(set(self.taxid))
        self.taxid_to_label = {taxid: i for i, taxid in enumerate(unique_taxids)}
        self.label_to_taxid = {i: taxid for taxid, i in self.taxid_to_label.items()}

        if cache_lineage:
            self.taxidLineage.populate(self.taxid, ['superkingdom'])

    # def get_sample_weight(self, taxid):
    #     weight = 0
    #     ranks = self.taxidLineage.get_ranks(taxid, ranks=['superkingdom'])
    #     superkingdom = ranks['superkingdom'][1]
    #     # calc sample weight
    #     weight += self.weight_classes['superkingdom'].get(superkingdom,
    #                                                       self.weight_classes['superkingdom']['unknown'])
    #     # weight += self.weight_classes['kingdom'].get(kingdom,self.weight_classes['kingdom']['unknown'])
    #     # weight += self.weight_classes['family'].get(family,self.weight_classes['family']['unknown'])
    #     return weight

    # def __len__(self):
    #     return len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq, taxid, _ = self.dataset[idx]  # 获取超界类别
        encoded_seq = self.tokenizer.encode(seq)
        encoded_seq = encoded_seq[:self.seq_len] + [self.tokenizer.pad_token_id] * (self.seq_len - len(encoded_seq))
        # 将taxid映射到整数标签
        label = self.taxid_to_label[taxid]
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)  # 返回超界类别

class MambaTax(L.LightningModule):
    def __init__(self, train_dataset, valid_dataset, n_layer, d_model, seq_len,
                 lr, lr_scheduler_factor, weight_decay,
                 batch_size_train, batch_size_val, gpu_cnt):
        super().__init__()
        self.classes = ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']
        self.training_entries = train_dataset
        self.validation_entries = valid_dataset

        # 合并训练和验证数据集中的taxids
        self.seq, self.taxid, self.superkingdom = zip(*train_dataset)
        self.seq_val, self.taxid_val, self.superkingdom_val = zip(*valid_dataset)

        # 获取所有唯一的taxids
        all_unique_taxids = sorted(set(self.taxid + self.taxid_val))

        # 创建统一的taxid到label和label到taxid的映射
        self.taxid_to_label = {taxid: i for i, taxid in enumerate(all_unique_taxids)}
        self.label_to_taxid = {i: taxid for taxid, i in self.taxid_to_label.items()}
        self.taxid_lineage = TaxidLineage()
        self.tokenizer = DNATokenizer()

        mamba_config = MambaConfig(n_layer=n_layer, d_model=d_model,
                                   vocab_size=self.tokenizer.vocab_size(),
                                   ssm_cfg={}, rms_norm=True, residual_in_fp32=True,
                                   fused_add_norm=True, pad_vocab_size_multiple=1)
        self.mambaDNA = MambaLMHeadModel(mamba_config)

        self.mambaDNA.lm_head = torch.nn.Linear(self.mambaDNA.config.d_model, len(all_unique_taxids))

        # self.classifier_head = nn.Linear(d_model, len(self.training_entries))

        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len = seq_len
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.weight_decay = weight_decay
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        self.train_accuracy = MulticlassAccuracy(num_classes=len(self.training_entries), average='micro')
        self.val_accuracy = MulticlassAccuracy(num_classes=len(self.validation_entries), average='micro')
        self.train_perplexity = FastPerplexity()
        self.val_perplexity = FastPerplexity()

        self.validation_outputs = []

        self.save_hyperparameters(ignore=['mambaDNA'])

    def forward(self, inpts):
        outputs = self.mambaDNA(inpts)
        logits = outputs.logits[:, -1, :]
        return logits

    def create_dataloader(self, dataset, batch_size, shuffle):
        classes = [entry[2] for entry in dataset]
        weights = {
            'superkingdom': {c: 1 / (len([yi for yi in classes if yi == c]) / len(classes)) for c in self.classes}}
        dataset = CustomGenomeDataset(dataset, self.tokenizer, self.seq_len, weights)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)

    def train_dataloader(self):
        return self.create_dataloader(self.training_entries, self.batch_size_train, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.validation_entries, self.batch_size_val, shuffle=False)

    def training_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast():
            inpts, trgts = batch
            outpts = self(inpts)
            loss = self.loss_fn(outpts, trgts)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        preds = torch.argmax(outpts, dim=1)
        predicted_taxids = [self.label_to_taxid[label.item()] for label in preds.cpu().numpy()]
        true_taxids_lable = trgts.cpu().numpy()

        true_taxids = [self.label_to_taxid[label] for label in true_taxids_lable]

        # 计算超界的准确率
        predicted_superkingdoms = [self.taxid_lineage.get_ranks(taxid).get('superkingdom', (None, 'unknown'))[1] for
                                   taxid in predicted_taxids]
        true_superkingdoms = [self.taxid_lineage.get_ranks(taxid).get('superkingdom', (None, 'unknown'))[1] for taxid in
                              true_taxids]

        correct_predictions = sum(p == t for p, t in zip(predicted_superkingdoms, true_superkingdoms))
        superkingdom_acc = correct_predictions / len(true_taxids)
        self.log("train_superkingdom_acc", superkingdom_acc, on_step=False, on_epoch=True)

        self.log("train_superkingdom_acc_step", superkingdom_acc, on_step=False, on_epoch=True)
        self.log("train_loss_step", loss, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        # 获取记录的平均损失
        epoch_loss = self.trainer.logged_metrics.get("train_loss_step", 0)

        # 获取记录的超界准确率
        epoch_superkingdom_acc = self.trainer.logged_metrics.get("train_superkingdom_acc_step", 0)

        # 记录这些值（如果需要）
        self.log("train_loss_epoch", epoch_loss, on_step=False, on_epoch=True)
        self.log("train_superkingdom_acc_epoch", epoch_superkingdom_acc, on_step=False, on_epoch=True)

        print(f"Epoch finished - Avg loss: {epoch_loss:.4f}, Avg superkingdom accuracy: {epoch_superkingdom_acc:.4f}")

    def validation_step(self, batch, batch_idx):
        inpts, trgts = batch
        outpts = self(inpts)
        loss = self.loss_fn(outpts, trgts)

        preds = torch.argmax(outpts, dim=1)
        predicted_taxids = [self.label_to_taxid[label.item()] for label in preds.cpu()]

        # 收集每一步的结果
        self.validation_outputs.append({"val_loss": loss, "val_preds": predicted_taxids, "val_trgts": trgts.tolist()})
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if self.validation_outputs:  # 确保列表不为空
            # 聚合所有批次的预测和真实标签
            all_preds = [item for sublist in [x['val_preds'] for x in self.validation_outputs] for item in sublist]
            all_trgts_labels = [item for sublist in [x['val_trgts'] for x in self.validation_outputs] for item in
                                sublist]
            ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            all_trgts = [self.label_to_taxid[label] for label in all_trgts_labels]

            # 计算每个分类级别上的准确率
            accuracy_per_rank = {}
            for rank in ranks:
                predicted_ranks = [self.taxid_lineage.get_ranks(taxid).get(rank, (None, 'unknown'))[1] for taxid in
                                   all_preds]
                true_ranks = [self.taxid_lineage.get_ranks(taxid).get(rank, (None, 'unknown'))[1] for taxid in
                              all_trgts]

                correct_predictions = sum(p == t for p, t in zip(predicted_ranks, true_ranks))
                accuracy = correct_predictions / len(all_trgts)

                accuracy_per_rank[rank] = accuracy

            # 记录每个分类级别上的准确率
            for rank, acc in accuracy_per_rank.items():
                self.log(f"val_acc_{rank}", acc)
                print(f"Accuracy at {rank} level: {acc}")

            # 清空缓存的输出，为下一个epoch做准备
            self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.mambaDNA.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                  factor=self.lr_scheduler_factor)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}

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