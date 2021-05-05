# coding: utf-8

import os
import numpy as np
from utils import *
from softmax_regression import Model

if __name__ == '__main__':
    if not os.path.exists('./dataset'):
        current_path = os.getcwd()           # 获得当前路径
        os.mkdir(current_path + '/dataset')  # 添加dataset路径
        path = './sentiment-analysis-on-movie-reviews/train.tsv'  # 原始数据所在路径
        generate_dataset(path)  # 生成数据集：训练集、开发集和测试集（train.txt、dev.txt和test.txt）

    train_path = './dataset/train.txt'
    vocab, data = build_vocab(train_path)
    feature = feature_extraction(vocab, data)

    model = Model()

    for x, y in data_iter(feature):
        x = np.asarray(x)
        y = np.asarray(y)
        one = np.ones((x.shape[0], 1))
        x = np.concatenate((x, one), axis=1)
        y_hat = model.forward(x)
        