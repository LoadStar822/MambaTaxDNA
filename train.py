#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/3 10:31
# @Author  : Qinzhong Tian
# @File    : train.py
# @Software: PyCharm
import numpy as np
import torch
import argparse
import lightning as L
from Bio import SeqIO
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from modules.MambaTaxSuper import MambaTaxSuper
from lightning.pytorch.callbacks import RichProgressBar
import sys
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

from modules.MambaTest import MambaTest

torch.set_float32_matmul_precision('medium')


def train_model():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./datatest/')
    args = parser.parse_args()
    data_path = args.data_path
    batch_size = args.batch_size

    def load_data_from_fasta(file_path):
        sequences, taxids = [], []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
            # 假设每个描述符是以'>taxid label'格式
            taxid = record.description.split(' ')[0]
            taxids.append(taxid)
        return sequences, taxids

    f_train, s_train = load_data_from_fasta(data_path + 'train_top_1_percent.fa')

    # 将训练数据分割为训练集和验证集（5%作为验证集）
    f_train, f_val, s_train, s_val = train_test_split(
        f_train, s_train, test_size=0.05, random_state=42)

    # 将数据组合成(train_data, val_data)元组的形式
    train_data = list(zip(f_train, s_train))
    val_data = list(zip(f_val, s_val))

    mambaDNATax = MambaTest(train_dataset=train_data,
                                valid_dataset=val_data,
                                n_layer=3, d_model=64, seq_len=502,
                                lr=1e-6, lr_scheduler_factor=0.85, weight_decay=0.02,
                                batch_size_train=batch_size, batch_size_val=batch_size, gpu_cnt=1)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_top_k=1, monitor="val_acc_superkingdom",
                                          mode="max")
    early_stop_callback = EarlyStopping(monitor="val_acc_superkingdom", min_delta=0.00, patience=args.patience,
                                        verbose=False,
                                        mode="max")
    logger = TensorBoardLogger("tb_logs", name="mamba_model")

    current_time = datetime.now().strftime("%Y%m%d%H%M")
    log_file_name = f"{current_time}.log"
    log_file_path = os.path.join('logs', log_file_name)
    trainer = L.Trainer(max_epochs=args.epochs,
                        accelerator='gpu', devices=1,
                        # logger=logger,
                        callbacks=[checkpoint_callback, early_stop_callback])

    # model = MambaTaxSuper.load_from_checkpoint(checkpoint_path='checkpoints/epoch=3-step=6308-v1.ckpt')
    trainer.fit(mambaDNATax)
    # trainer.fit(model)


if __name__ == "__main__":
    train_model()
