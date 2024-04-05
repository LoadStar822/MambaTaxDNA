#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/3 15:50
# @Author  : Qinzhong Tian
# @File    : KmerEncoder.py
# @Software: PyCharm
import numpy as np
from itertools import product
from numba import njit

class KmerEncoder:
    def __init__(self, k=3, stride=1, pad=True, to_upper=True):
        self.k = k
        self.stride = stride
        self.pad = pad
        self.to_upper = to_upper
        self.kmer_index = self.generate_kmer_index()  # 生成k-mer索引字典
        self.kmer_index_list = list(self.kmer_index)

    def generate_kmer_index(self):
        """生成k-mer索引字典，每个k-mer映射到唯一的索引。"""
        bases = ['A', 'C', 'G', 'T']
        kmers = [''.join(p) for p in product(bases, repeat=self.k)]
        kmer_index = {kmer: i for i, kmer in enumerate(kmers)}
        return kmer_index

    def encode_sequence(self, sequence):
        """将DNA序列编码为k-mers列表。"""
        return self.generate_kmers(sequence)

    def fasta2kmer(self, data):
        """编码整个数据集，每个序列转换为k-mers列表。"""
        encoded_data = []
        for sequence, taxid, category in data:
            encoded_sequence = self.encode_sequence(sequence)
            encoded_data.append((encoded_sequence, taxid, category))
        return encoded_data

    @staticmethod
    @njit(parallel=True)
    def generate_kmers_njit(sequence, k, stride, pad_char):
        """使用Numba优化的k-mer生成函数。"""
        kmers = []
        for i in np.arange(0, len(sequence) - k + 1, stride):
            kmers.append(sequence[i:i + k])
        if len(sequence) % k != 0 and pad_char:
            padding_length = k - (len(sequence) % k)
            kmers.append(sequence[-(len(sequence) % k):] + pad_char * padding_length)
        return kmers

    def generate_kmers(self, sequence):
        """为给定序列生成k-mers，考虑步长、填充和大小写转换。"""
        if self.to_upper:
            sequence = sequence.upper()

        # 替换非ATCG字符为'N'
        valid_bases = set('ATCG')
        sequence = ''.join([base if base in valid_bases else 'N' for base in sequence])

        pad_char = 'N' if self.pad else ''
        return self.generate_kmers_njit(sequence, self.k, self.stride, pad_char)

