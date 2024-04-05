#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 17:23
# @Author  : Qinzhong Tian
# @File    : Test.py
# @Software: PyCharm
import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.MambaTax import CustomGenomeDataset
from train import MambaTax
from utils.Tokenizer import DNATokenizer
from utils.tax_entry import TaxidLineage

# 步骤1: 加载模型
model = MambaTax.load_from_checkpoint(checkpoint_path='checkpoints/epoch=5-step=156.ckpt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data_path = './datatest/'
f_test = np.load(data_path + 'f_test.npy')
s_test = np.load(data_path + 's_test.npy')
c_test = np.load(data_path + 'c_test.npy')
test_data = list(zip(f_test, s_test, c_test))

# 步骤2: 准备数据
# 如果您有一个CustomGenomeDataset类似的数据加载类，可以这样加载测试数据
tokenizer = DNATokenizer()
test_dataset = CustomGenomeDataset(test_data, tokenizer, 502)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
taxid_lineage = TaxidLineage()

# 步骤3: 进行测试
model.eval()  # 设置为评估模式
predictions = []
for batch in test_dataloader:
    seqs, _ = batch
    seqs = seqs.to(device)  # 确保输入数据也在正确的设备上
    with torch.no_grad():
        outputs = model(seqs)
        predicted_labels = torch.argmax(outputs, dim=1)
        predicted_taxids = [model.label_to_taxid[label.item()] for label in predicted_labels]
        predictions.extend(predicted_taxids)

# 现在，predictions列表包含了对测试数据的预测结果（taxid）
true_taxids = s_test.tolist()
ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
accuracy_per_rank = {}
# 对每个级别进行迭代
for rank in ranks:
    predicted_ranks = [taxid_lineage.get_ranks(taxid).get(rank, (None, 'unknown'))[1] for taxid in predictions]
    true_ranks = [taxid_lineage.get_ranks(taxid).get(rank, (None, 'unknown'))[1] for taxid in true_taxids]

    # 计算当前级别的准确率
    correct_predictions = sum(p == t for p, t in zip(predicted_ranks, true_ranks))
    accuracy = correct_predictions / len(true_taxids)

    # 存储结果
    accuracy_per_rank[rank] = accuracy

# 输出所有级别的准确率
for rank, acc in accuracy_per_rank.items():
    print(f"Accuracy at {rank} level: {acc}")
