# coding: utf-8

import random
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from utils import *
from model import logistic_regression as lr

############### 数据集路径 #################
train_path = './data_sentiment/train.csv'
dev_path = './data_sentiment/dev.csv'
test_path = './data_sentiment/test.csv'
########3######    停用词    #################
stop_words = stopwords.words('english')
# 模型参数配置
config = lr.Config()
# 清洗数据：将字符串转换成小写、去掉标点符号和停用词
train_data = clean_data(train_path, stop_words)
dev_data = clean_data(dev_path, stop_words)
test_data = clean_data(test_path, stop_words)
# 根据训练集生成词典（去频率前5000的词，将每个词都映射到一个唯一的id）
vocab = build_vocab(train_data, n_gram=config.n_gram)

config.vocab = vocab
# 提取数据特征（word count、tf-idf）
train_feature = feature_extraction(train_data, vocab)
dev_feature = feature_extraction(dev_data, vocab)
test_feature = feature_extraction(test_data, vocab)

# 构建模型
model = lr.Model(config)
# 训练模型
train(config, model, train_feature, dev_feature)

L = len(test_feature)
test_true_nums = 0
test_loss = 0.
for x, y in test_feature:
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    y_hat = sigmoid(np.matmul(model.weight.T, x))
    y_pred = int(y_hat[0][0] > 0.5)
    test_true_nums += int((y_pred == y))
    test_loss += binary_cross_entropy_loss(model.weight, y, y_hat)

print(test_true_nums/L, test_loss/L)