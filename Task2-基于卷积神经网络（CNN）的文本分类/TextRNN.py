# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.model_name = 'TextRNN'  # 模型名称
        self.maxLen = 32    # 句子的最大长度
        self.vocab = None   # 词表，在运行时赋值
        self.classes = 3    # 文本类别数量
        self.embedding_dim = 100  # 词向量维度
        self.hiddenSize = 128     # 隐藏层神经元数量
        self.save_path = './model/' + self.model_name + '.ckpt'  # 模型保存路径
        self.pretrain_path = './pretrained_wordvector/pretrained_embedding.npy'  # 预训练词向量路径
        self.pretrained = np.load(self.pretrain_path)  # 加载预训练词向量
        # self.pretrained = 'random'  # 假如没有预训练词向量，就随机初始化词向量
        self.batchSize = 128       # batch_size
        self.learningRate = 1e-4  # 学习率

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        torch.manual_seed(123)
        if config.pretrained == 'random':
            # 假如没有预训练词向量，就随机初始化词向量
            self.embedding = nn.Embedding(len(config.vocab), config.embedding_dim)
        else:
            # 如果有预训练词向量，那么就先生成随机的词向量，然后将预训练词向量复制到这个随机生成的预训练词向量中
            self.embedding = nn.Embedding(len(config.vocab), config.embedding_dim)
            self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained))
            self.embedding.weight.requires_grad = True  # 随模型训练进行微调，False-保持不变
        self.rnn = nn.RNN(config.embedding_dim, config.hiddenSize, batch_first=True, num_layers=2, bidirectional=False,
                          dropout=0.5)
        self.seq_net = nn.Sequential(
            nn.Linear(config.hiddenSize, 64),
            nn.ReLU(),
            nn.Linear(64, config.classes)
        )
        #  self.fc = nn.Linear(config.hiddenSize, config.classes)

    def forward(self, x):
        x = self.embedding(x)
        _, hn = self.rnn(x)  # RNN
        # hn = hn.squeeze()
        hn = hn[-1]
        out = self.seq_net(hn)
        return out

class Attention():
    def __init__(self, X, q):
        super(Attention, self).__init__()
        Dx = X.size()[-1]
        alpha = F.softmax(torch.matmul(X, q) / torch.sqrt(Dx), dim=1).permute(0, 2, 1)
        attVec = torch.matmul(alpha, X).squeeze()
        return attVec


if __name__ == '__main__':
    ### Pytorch中RNN、LSTM、GRU的使用详解参考：https://blog.csdn.net/lkangkang/article/details/89814697
    # 输入特征维度5，输出维度10, 层数2
    rnn = torch.nn.RNN(5, 10, 3, batch_first=True)
    # seq长度4，batch_size=2
    input = torch.randn(2, 4, 5)
    output, hn = rnn(input)
    print(output.size(), hn.size())
    q = torch.randn(10, 1)
    alpha = F.softmax(torch.matmul(output, q), dim=1).permute(0, 2, 1)
    print(alpha.size())
    print(alpha)
    attVec = torch.matmul(alpha, output)
    print(attVec.size())
    print(attVec)