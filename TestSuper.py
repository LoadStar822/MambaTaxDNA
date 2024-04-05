#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 17:23
# @Author  : Qinzhong Tian
# @File    : Test.py
# @Software: PyCharm
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader

from modules.MambaTaxSuper import CustomGenomeDataset, MambaTaxSuper
from utils.Tokenizer import DNATokenizer
from utils.tax_entry import TaxidLineage

# 步骤1: 加载模型
model = MambaTaxSuper.load_from_checkpoint(checkpoint_path='checkpoints/epoch=9-step=15770.ckpt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data_path = './datatest/'


# 从FASTA文件中读取序列和taxid
def load_data_from_fasta(file_path):
    sequences, taxids = [], []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        # 假设每个描述符是以'>taxid label'格式
        taxid = record.description.split(' ')[0]
        taxids.append(taxid)
    return sequences, taxids


f_test, s_test = load_data_from_fasta(data_path + 'test.fa')

test_data = list(zip(f_test, s_test))

# 步骤2: 准备数据
# 如果您有一个CustomGenomeDataset类似的数据加载类，可以这样加载测试数据
tokenizer = DNATokenizer()
test_dataset = CustomGenomeDataset(test_data, tokenizer, 502, model.superkingdom_to_label, model.phylum_to_label)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
taxid_lineage = TaxidLineage()

# 步骤3: 进行测试
model.eval()  # 设置为评估模式
predictions_super = []
predictions_phylum = []
for batch in test_dataloader:
    seqs, _, _ = batch
    seqs = seqs.to(device)  # 确保输入数据也在正确的设备上
    with torch.no_grad():
        outputs = model(seqs)
        predicted_super_label = torch.argmax(outputs[0], dim=1)
        predicted_phylum_label = torch.argmax(outputs[1], dim=1)

        predicted_super = [model.label_to_superkingdom[label.item()] for label in predicted_super_label]
        predicted_phylum = [model.label_to_phylum[label.item()] for label in predicted_phylum_label]
        predictions_super.extend(predicted_super)
        predictions_phylum.extend(predicted_phylum)

# 现在，predictions列表包含了对测试数据的预测结果（taxid）
true_taxids = s_test
true_super = [taxid_lineage.get_ranks(taxid)['superkingdom'][1] for taxid in true_taxids]
ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
accuracy_per_rank = {}
# 对每个级别进行迭代
true_phylum = [taxid_lineage.get_ranks(taxid)['phylum'][1] for taxid in true_taxids]

correct_predictions_super = sum(p == t for p, t in zip(predictions_super, true_super))
accuracy_super = correct_predictions_super / len(true_taxids)
correct_predictions_phylum = sum(p == t for p, t in zip(predictions_phylum, true_phylum))
accuracy_phylum = correct_predictions_phylum / len(true_taxids)

print(f"Accuracy at superkingdom level: {accuracy_super}")
print(f"Accuracy at phylum level: {accuracy_phylum}")