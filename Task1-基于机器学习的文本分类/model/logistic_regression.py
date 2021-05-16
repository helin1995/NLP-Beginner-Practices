import numpy as np

class Config(object):
    def __init__(self):
        self.vocab = {}  # 训练集上出现的频率前k个词
        self.n_gram = 1  # n_gram
        self.learning_rate = 1e-4  # 学习率
        self.lamda = 1e-5      # 正则化系数
        self.tolerance = 1e-4  # early stop的阈值，当开发集上相邻两个loss的差距小于这个阈值时，可以认为开发集的loss不再下降，就停止训练。


class Model(object):
    def __init__(self, config):
        self.weight = np.zeros((len(config.vocab), 1))