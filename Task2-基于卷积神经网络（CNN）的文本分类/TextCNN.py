# coding: utf-8

import torch
import numpy as np
import torch.nn as nn

class Config(object):
    def __init__(self):
        self.model_name = 'TextCNN'  # 模型名称
        self.maxLen = 64    # 句子的最大长度
        self.vocab = None   # 词表，在运行时赋值
        self.embedding_dim = 100  # 词向量维度
        self.save_path = './model/' + self.model_name + '.ckpt'  # 模型保存路径
        self.pretrain_path = './pretrained_wordvector/pretrained_embedding.npy'  # 预训练词向量路径
        self.pretrained = np.load(self.pretrain_path)  # 加载预训练词向量
        # self.pretrained = 'random'  # 假如没有预训练词向量，就随机初始化词向量
        self.batchSize = 64       # batch_size
        self.learningRate = 1e-3  # 学习率

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        torch.manual_seed(1)
        if config.pretrained == 'random':
            # 假如没有预训练词向量，就随机初始化词向量
            self.embedding = nn.Embedding(len(config.vocab), config.embedding_dim)
        else:
            # 如果有预训练词向量，那么就先生成随机的词向量，然后将预训练词向量复制到这个随机生成的预训练词向量中
            self.embedding = nn.Embedding(len(config.vocab), config.embedding_dim)
            self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained))
            self.embedding.weight.requires_grad = True  # 随模型训练进行微调，False-保持不变
        # 卷积核池化层，共有三个，窗口大小分别为2， 3， 4；卷积核2个。
        self.conv2 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=2, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=config.maxLen - 2 + 1)
        self.conv3 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=2, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=config.maxLen - 3 + 1)
        self.conv4 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=2, kernel_size=4, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=config.maxLen - 4 + 1)
        self.fc = nn.Linear(6, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # x=[batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embedding]
        x = x.permute(0, 2, 1)  # [batch, embedding, seq_len]
        x2 = self.conv2(x)  # [batch, out_channels, seq_len-kernel_size+1]
        x2 = self.pool2(x2)  # [batch, out_channels, 1]
        x3 = self.conv3(x)
        x3 = self.pool3(x3)
        x4 = self.conv4(x)
        x4 = self.pool4(x4)
        out = torch.cat([x2, x3, x4], dim=1)  # [batch, 6, 1]
        out = out.squeeze()  # [batch, 6]
        out = self.dropout(out)
        out = self.fc(out)  # [batch, 3]
        return out
