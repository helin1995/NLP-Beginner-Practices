# coding: utf-8

import numpy as np

# 程序参数配置
class Config(object):
    def __init__(self):
        self.vocab = dict()    # 根据训练集生成的词典，在运行时重新赋值
        self.feature_dims = 0  # 样本特征维度，在运行时赋值
        self.class_nums = 5    # 数据集标签，共5种
        self.batch_size = 128  # mini-batch的大小
        self.save_path = './model/weights'  # 模型参数保存路径
        self.learning_rate = 5e-2  # 模型学习率



class Model(object):
    def __init__(self, config):
        self.W = np.random.randn(config.class_nums, config.feature_dims)  # 模型权重
        self.b = np.random.randn(config.class_nums, 1)                    # 模型的偏置
        self.Weight = np.concatenate((self.W, self.b), axis=1)

    def softmax(self, x):
        '''
        利用softmax函数得到模型预测的样本类别概率分布
        :param x: W * x.T
        :return:
        '''
        # x: [5, 128]
        ex = np.exp(x)  # [5, 128]
        row, col = ex.shape
        ex_sum = np.sum(ex, axis=0)  # [1, 128]
        for i in range(col):
            for j in range(row):
                ex[j][i] /= ex_sum[i]
        return ex

    def forward(self, x):
        '''
        模型预测的结果：res=softmax(W * x.T)
        :param x: mini-batch数据
        :return: 模型预测的结果：mini-batch中每个样本的概率分布
        '''
        res = np.matmul(self.Weight, x.T)
        res = self.softmax(res)
        return res
