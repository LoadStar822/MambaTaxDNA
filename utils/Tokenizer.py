#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 12:38
# @Author  : Qinzhong Tian
# @File    : Tokenizer.py
# @Software: PyCharm
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class DNATokenizer(PreTrainedTokenizer):
    def __init__(self, model_max_length=512, **kwargs):
        # 设置模型最大长度默认值为1Gbp
        self.model_max_length = model_max_length
        # 定义各种特殊token
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        # 为complement操作添加额外的[SWAP] token
        characters = ['A', 'C', 'G', 'T', 'N', '[SWAP]']

        # 调用基类构造函数初始化tokenizer
        super().__init__(
            bos_token=bos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side="left",
            **kwargs,
        )
        self.add_tokens(characters)

    # 返回词汇表大小
    def vocab_size(self) -> int:
        return len(self.added_tokens_encoder)

    # 获取词汇表
    def get_vocab(self):
        return self.added_tokens_encoder

    # 将token转换为字符串
    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    # 清理序列，将非AGCT字符替换为N
    def clean_sequence(self, sequence):
        return ''.join([char if char in ['A', 'C', 'G', 'T'] else 'N' for char in sequence])

    # 编码DNA序列
    def encode(self, text, **kwargs):
        clean_text = self.clean_sequence(text)
        return [self.token_to_id(token) for token in clean_text]

    # 解码为DNA序列
    def decode(self, token_ids, **kwargs):
        return ''.join(self.id_to_token(token_id) for token_id in token_ids)

    # 将token转换为ID
    def token_to_id(self, token):
        return self.added_tokens_encoder.get(token, self.unk_token_id)

    # 将ID转换为token
    def id_to_token(self, token_id):
        return {v: k for k, v in self.added_tokens_encoder.items()}.get(token_id, self.unk_token)