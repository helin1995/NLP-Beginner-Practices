# coding: utf-8

import torch
from utils import *
from TextCNN import Config

if __name__ == '__main__':
    token_pattern = r'\b[A-Za-z][A-Za-z]+\b'  # 正则表达式：只取每篇文档中两端是字母的词
    tokenizer = lambda x: x.split()
    trainData = load_data('./dataset/train.csv', token_pattern)
    vocab = build_vocab(trainData)
    config = Config()
    train = generate_data(trainData, vocab, config)
