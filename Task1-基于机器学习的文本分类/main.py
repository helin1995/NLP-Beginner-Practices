# coding: utf-8

import os
import numpy as np
from utils import *
from softmax_regression import Model, Config

if __name__ == '__main__':
    if not os.path.exists('./dataset'):
        # 检查是否存在dataset文件夹，如果不存在，那么就根据原始训练数据生成训练集、开发集和测试集
        current_path = os.getcwd()           # 获得当前路径
        os.mkdir(current_path + '/dataset')  # 添加dataset路径
        path = './sentiment-analysis-on-movie-reviews/train.tsv'  # 原始数据所在路径
        generate_dataset(path)  # 生成数据集：训练集、开发集和测试集（train.txt、dev.txt和test.txt）

    config = Config()  # 生成模型配置对象

    train_path = './dataset/train.txt'      # 训练集路径
    vocab, data = build_vocab(train_path)   # 根据训练集生成词典和训练数据（词列表）
    config.vocab = vocab                    # 赋值配置文件中的词典
    config.feature_dims = len(vocab)        # 样本特征维度，在运行时赋值
    train_feature = feature_extraction(config.vocab, data, flag=True)    # 根据词典和训练集进行特征提取，用于模型的输入
    # train_iter = data_iter(train_feature, config.batch_size)  # 训练集mini-batch

    dev_path = './dataset/dev.txt'          # 开发集路径
    dev_data = data_clean(dev_path)         # 对开发集中的数据进行数据清洗，包括：去掉标点符号、转换成小写、去掉停用词等
    dev_feature = feature_extraction(config.vocab, dev_data, flag=True)  # 根据词典和开发集进行特征提取，用于模型的输入
    # dev_iter = data_iter(dev_feature, config.batch_size)      # 开发集mini-batch

    test_path = './dataset/test.txt'        # 测试集路径
    test_data = data_clean(test_path)       # 对测试集中的数据进行数据清洗，包括：去掉标点符号、转换成小写、去掉停用词等
    test_feature = feature_extraction(config.vocab, test_data, flag=True)  # 根据词典和测试集进行特征提取，用于模型的输入
    # test_iter = data_iter(test_feature, config.batch_size)      # 测试集mini-batch

    model = Model(config)  # 构建模型，并对模型参数初始化

    train(config, model, train_feature, dev_feature)  # 训练模型
