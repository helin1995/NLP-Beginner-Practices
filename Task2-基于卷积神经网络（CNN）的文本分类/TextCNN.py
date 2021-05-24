# coding: utf-8

import torch
import torch.nn as nn

class Config(object):
    def __init__(self):
        self.maxLen = 64   #  句子的最大长度
        self.vocab = None   # 词表，在运行时赋值
        self.embedding_dim = 300
        self.batchSize = 32
        self.learningRate = 1e-3

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(len(config.vocab), config.embedding_dim)
        self.conv2 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=2, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=config.maxLen - 2 + 1)
        self.conv3 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=2, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=config.maxLen - 3 + 1)
        self.conv4 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=2, kernel_size=4, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=config.maxLen - 4 + 1)
        self.fc = nn.Linear(6, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embedding]
        x = x.permute(0, 2, 1)  # 32x300x64
        x2 = self.conv2(x)  # 32x2x63
        x2 = self.pool2(x2)  # 32x2x1
        x3 = self.conv3(x)  # 32x2x62
        x3 = self.pool3(x3)  # 32x2x1
        x4 = self.conv4(x)  # 32x2x61
        x4 = self.pool4(x4)  # 32x2x1
        out = torch.cat([x2, x3, x4], dim=1)  # 32x6x1
        out = out.squeeze()  # 32x6
        out = self.dropout(out)
        out = self.fc(out)  # 32x3
        return out
