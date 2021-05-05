# coding: utf-8

import os
import random
import string
import pickle as pkl
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

MAX_VOCAB_SIZE = 10000  # 词表最大长度
MIN_FREQ = 1            # 单词最小出现次数

def statistic(path):
    '''
    统计原始数据文件（.tsv）中多少条数据，数据共多少类别，以及样本最大的长度。
    '''
    read_data = pd.read_table(path)
    all_items, _ = read_data.shape  # 样本总数
    max_length = 0  # 样本最大长度
    label = []      # 样本标签
    occu_dict = {}  # 原始数据中出现过的词
    for i in range(all_items):
        sentence = (read_data.loc[i]['Phrase']).lower()
        if len(sentence.split()) > max_length:
            max_length = len(sentence.split())
        label.append(read_data.loc[i]['Sentiment'])
        for word in sentence.split():
            occu_dict[word] = occu_dict.get(word, 0) + 1
    return all_items, len(set(label)), max_length, occu_dict

def generate_dataset(path, test_rate=0.3):
    '''
    根据提供的数据生成数据集：训练集、开发集和测试集
    path: 原始数据所在路径
    test_rate: 测试集占原始数据的比例
    return:
    '''
    read_data = pd.read_table(path)
    rows, _ = read_data.shape
    data = []
    for i in range(rows):
        sentence = read_data.loc[i]['Phrase']  # 文本内容
        label = str(read_data.loc[i]['Sentiment'])  # 文本对应的标签
        data.append([sentence, label])
    random.shuffle(data)  # 打乱data的顺序
    train = data[int(len(data)*test_rate):]
    test_set = data[:int(len(data)*test_rate)]  # 测试集
    dev_set = train[:int(len(train)*test_rate)]  # 开发集
    train_set = train[int(len(train)*test_rate):]  # 训练集
    with open('./dataset/train.txt', 'w', encoding='utf-8') as tra, open('./dataset/dev.txt', 'w', encoding='utf-8') as dev, open('./dataset/test.txt', 'w', encoding='utf-8') as test:
        for item in train_set:
            tra.write(item[0] + '|||' + item[1] + '\n')
        for item in dev_set:
            dev.write(item[0] + '|||' + item[1] + '\n')
        for item in test_set:
            test.write(item[0] + '|||' + item[1] + '\n')

def data_clean(data_path):
    '''
    数据清洗：去掉样本中的标点符号、停用词
    :param data: 数据文本存放路径
    :return: 清洗后的数据
    '''
    print('正在进行数据清洗！')
    with open(data_path, 'r', encoding='utf-8') as fr:
        data = []
        for line in fr.readlines():
            sentence, label = line.strip().split('|||')
            sentence = sentence.lower()  # 将文本中的单词变为小写
            new_sentence = []
            for word in sentence.split(' '):
                if (word not in string.punctuation) and (word not in stopwords.words('english')):
                    new_sentence.append(word)
            data.append([new_sentence, int(label)])
    print('数据清洗完成！')
    return data

def build_vocab(path):
    '''
    根据data构建词典
    :param data:
    :return:
    '''
    data = data_clean(path)
    print('开始生成词典！')
    vocab = dict()
    for item in data:
        sentence, _ = item
        for word in sentence:
            vocab[word] = vocab.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] >= MIN_FREQ], key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE]
    vocab = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    with open('vocabulary/vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f, protocol=pkl.HIGHEST_PROTOCOL)
    print('已生成好词典！')
    return vocab, data


def feature_extraction(vocab, data):
    feature = []
    for sentence, label in data:
        new_data = [0] * len(vocab)
        for word in sentence:
            if word in vocab.keys():
                new_data[vocab[word]] = 1
        feature.append([new_data, label])
    return feature

def data_iter(d):
    batch_size = 128
    n = len(d) // batch_size
    for i in range(n):
        X = []
        Y = []
        for j in range(i*batch_size, (i+1)*batch_size):
            x, y = d[j]
            X.append(x)
            Y.append(y)
        yield X, Y
    X = []
    Y = []
    for j in range(n*batch_size, len(d)):
        x, y = d[j]
        X.append(x)
        Y.append(y)
    yield X, Y
