# coding: utf-8
import torch
from torch import nn
class Config(object):
    def __init__(self):
        self.vocab_size = 5000  # 词典大小
        self.n_gram = 1         # n_gram
        self.learning_rate = 1e-2  # 学习率
        self.classes = 3        # 样本类别总数
        self.tolerance = 1e-3   # 相邻两次dev_loss的差距阈值，小于则结束训练

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.layer = nn.Linear(config.vocab_size, config.classes)  # (5000, 3)

    def forward(self, x):
        out = self.layer(x)
        return torch.unsqueeze(out, 0)