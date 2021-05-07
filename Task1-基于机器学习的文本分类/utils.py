# coding: utf-8

import os
import random
import string
import math
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
        sentence = read_data.loc[i]['Phrase']       # 文本内容
        label = str(read_data.loc[i]['Sentiment'])  # 文本对应的标签
        data.append([sentence, label])
    random.shuffle(data)  # 打乱data的顺序
    train = data[int(len(data)*test_rate):]
    test_set = data[:int(len(data)*test_rate)]     # 测试集
    dev_set = train[:int(len(train)*test_rate)]    # 开发集
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

def term_frequency(word, sentence):
    times_of_word = 0
    for w in sentence:
        if w == word:
            times_of_word += 1
    return times_of_word / len(sentence)

def idf(word, data):
    word_times_in_data = 0
    for sentence, _ in data:
        if word in sentence:
            word_times_in_data += 1
    return math.log2(len(data) / (word_times_in_data + 1))

def tf_idf(vocab, data):
    feature = []
    idf_dict = {}
    nums = 1
    for sentence, label in data:
        new_data = [0] * len(vocab)
        for word in sentence:
            if word in vocab.keys():
                if word not in idf_dict.keys():
                    idf_val = idf(word, data)
                    new_data[vocab[word]] = term_frequency(word, sentence) * idf_val
                    idf_dict[word] = idf_val
                else:
                    new_data[vocab[word]] = term_frequency(word, sentence) * idf_dict[word]
        feature.append([new_data, label])
        print('[{}/{}]'.format(nums, len(data)))
        nums += 1
    return feature

def feature_extraction(vocab, data, flag=False, n_gram=1):
    '''
    特征抽取：将文本转换成向量（tf-idf、one-hot）
    :param vocab:
    :param data:
    :param flag: 是否采用tf-idf
    :param n_gram:
    :return:
    '''
    feature = []
    if flag:
        print('提取tf-idf特征')
        feature = tf_idf(vocab, data)
    else:
        for sentence, label in data:
            new_data = [0] * len(vocab)
            for word in sentence:
                if word in vocab.keys():
                    new_data[vocab[word]] = 1
            feature.append([new_data, label])
    return feature

def data_iter(d, batch_size):
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


def one_hot_encode(y, class_nums):
    '''
    将标签改为one-hot编码
    '''
    y_new = np.zeros((class_nums, len(y)))
    for idx, val in enumerate(y):
        y_new[int(val)][idx] = 1
    return y_new


def cross_entropy_loss(y_, y):
    '''
    交叉熵损失函数
    y_：模型预测值
    y：真实标签值
    '''
    N = y.shape[1]  # batch_size
    loss = 0.
    for i in range(N):
        for j in range(5):
            loss += -y[j][i] * np.log(y_[j][i])
    # loss = loss / N
    return (loss / N, loss)


def compute_accuracy(y_, y):
    N = len(y)
    correct_nums = 0
    for i in range(N):
        if y_[i] == y[i]:
            correct_nums += 1
    return (correct_nums / N, correct_nums)


def update_parameter(config, model, x, y, y_):
    lr = config.learning_rate
    N = x.shape[0]
    sigma = np.zeros((model.Weight.shape[0], model.Weight.shape[1]))
    for i in range(N):
        sigma += np.matmul((y.T[i] - y_.T[i]).reshape(-1, 1), x[i].reshape(1, -1))
    model.Weight = model.Weight + lr * sigma / N

def train(config, model, train_feature, dev_feature):
    total_batch = 1
    best_dev_accuracy = 0.
    flag = False
    last_improve = 0
    for epoch in range(20):
        print('epoch: [%d/%d]' % (epoch + 1, 20))
        for x, y in data_iter(train_feature, config.batch_size):
            x = np.asarray(x)
            y = np.asarray(y)
            one = np.ones((x.shape[0], 1))
            x = np.concatenate((x, one), axis=1)     # 将x变为增广矩阵
            y_hat = model.forward(x)                 # 使用模型得到每个样本的标签（类别）的概率分布
            y_pred_label = np.argmax(y_hat, axis=0)  # 使用argmax得到模型预测的结果
            accuracy, _ = compute_accuracy(y_pred_label, y)  # 计算mini-batch的准确率
            y = one_hot_encode(y, config.class_nums)         # 将样本标签转化为one-hot向量，用于使用交叉熵损失函数计算mini-batch的loss
            loss, _ = cross_entropy_loss(y_hat, y)           # 计算mini-batch loss
            update_parameter(config, model, x, y, y_hat)     # 更新参数
            if total_batch % 100 == 0:
                dev_accu, dev_loss = evaluate(config, model, dev_feature)
                if dev_accu > best_dev_accuracy:
                    best_dev_accuracy = dev_accu
                    improve = '*'
                    last_improve = total_batch
                    # save model
                    np.save(config.save_path, model.Weight, allow_pickle=True, fix_imports=True)
                else:
                    improve = ''
                print('train loss: {:.4f}\t train accuracy: {:.4f}\t dev accuracy: {:.4f}\t dev loss: {:.4f}\t{}'.format(loss, accuracy, dev_accu, dev_loss, improve))
            if total_batch - last_improve > 300:
                flag = True
                break
            total_batch += 1
        if flag:
            break


def evaluate(config, model, dev_feature):
    N = len(dev_feature)  # 开发集样本总数
    batch_correct_nums = 0.
    batch_loss = 0.
    for x, y in data_iter(dev_feature, config.batch_size):
        x = np.asarray(x)
        y = np.asarray(y)
        one = np.ones((x.shape[0], 1))
        x = np.concatenate((x, one), axis=1)  # 将x变为增广矩阵
        y_hat = model.forward(x)  # 使用模型得到每个样本的标签（类别）的概率分布
        y_pred_label = np.argmax(y_hat, axis=0)  # 使用argmax得到模型预测的结果
        _, correct_nums = compute_accuracy(y_pred_label, y)  # 计算mini-batch的准确率
        batch_correct_nums += correct_nums
        y = one_hot_encode(y, 5)  # 将样本标签转化为one-hot向量，用于使用交叉熵损失函数计算mini-batch的loss
        _, loss = cross_entropy_loss(y_hat, y)  # 计算mini-batch loss
        batch_loss += loss
    return (batch_correct_nums / N, batch_loss / N)