# coding: utf-8

import torch
import numpy as np
from torch import nn
from utils import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from model import softmax_regression as smr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 数据集路径
    train_path = './data_topic/train.csv'
    dev_path = './data_topic/dev.csv'
    test_path = './data_topic/test.csv'
    # 加载数据
    train_corpus, train_label = load_data(train_path)
    dev_corpus, dev_label = load_data(dev_path)
    test_corpus, test_label = load_data(test_path)
    # 设置CountVectorizer的参数
    token_pattern = r'\b[A-Za-z][A-Za-z]+\b'  # 正则表达式：只取每篇文档中两端是字母的词
    config = smr.Config()  # 模型参数配置
    stop_words = stopwords.words('english')   # 停用词
    # 生成CountVectorizer对象，用于计算每篇文档的词计数（word count）
    vectorizer = CountVectorizer(ngram_range=(config.n_gram, config.n_gram),
                                 stop_words=stop_words,
                                 token_pattern=token_pattern,
                                 max_features=config.vocab_size)
    # 生成TfidfTransformer对象，用于计算tf_idf值
    transformer = TfidfTransformer(smooth_idf=True)
    # 计算得到训练集的word count、tf_idf
    train_counts = vectorizer.fit_transform(train_corpus)
    X_train = transformer.fit_transform(train_counts).toarray()
    y_train = np.array(train_label)
    # 计算得到开发集的word count、tf_idf
    dev_counts = vectorizer.transform(dev_corpus)
    X_dev = transformer.fit_transform(dev_counts).toarray()
    y_dev = np.array(dev_label)
    # 计算得到测试集的word count、tf_idf
    test_counts = vectorizer.transform(test_corpus)
    X_test = transformer.fit_transform(test_counts).toarray()
    y_test = np.array(test_label)
    # 创建模型
    model = smr.Model(config)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    # 开发集上的初始损失：无穷大
    last_dev_loss = float('inf')

    train_history_loss = []
    dev_history_loss = []
    # 训练模型
    for epoch in range(200):
        # 打乱训练集
        index = np.random.permutation(range(len(X_train)))
        X_train = X_train[index]
        y_train = y_train[index]
        train_loss = 0.
        train_correct = 0
        print('[{}/{}]'.format(epoch+1, 200))
        for x, y in zip(X_train, y_train):
            x = torch.from_numpy(x).type(dtype=torch.FloatTensor)
            y = torch.from_numpy(np.array([y])).type(dtype=torch.LongTensor)
            # 计算样本x的预测值
            y_ = model(x)
            # 计算loss
            loss = criterion(y_, y)
            # 将优化其中的参数梯度归零
            optimizer.zero_grad()
            # 计算参数梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            train_loss += loss.detach().item()
            y_pred = torch.argmax(y_)
            if y_pred == y:
                train_correct += 1
        train_history_loss.append(train_loss / len(X_train))
        # 在开发集上进行评价
        dev_loss, dev_accu = evaluate(model, criterion, X_dev, y_dev)
        dev_history_loss.append(dev_loss)
        print_str = 'train loss: {:.4f} \t train accuracy: {:.4f} \t dev loss: {:.4f} \t dev accuracy: {:.4f}'
        print(print_str.format(train_loss / len(X_train), train_correct / len(X_train), dev_loss, dev_accu))
        # early stop
        if abs(dev_loss - last_dev_loss) < config.tolerance:
            break
        else:
            last_dev_loss = dev_loss
    # 训练完成后在测试集上进行测试
    test_loss = 0.
    test_correct = 0
    for x, y in zip(X_test, y_test):
        x = torch.from_numpy(x).type(dtype=torch.FloatTensor)
        y = torch.from_numpy(np.array([y])).type(dtype=torch.LongTensor)
        y_ = model(x)
        loss = criterion(y_, y)
        test_loss += loss.detach().item()
        y_pred = torch.argmax(y_)
        if y_pred == y:
            test_correct += 1
    print('test loss: {:.4f} \t test accuracy: {:.4f}'.format(test_loss / len(X_test), test_correct / len(X_test)))

    # 绘制loss曲线
    plt.plot(range(epoch+1), train_history_loss, 'b', label='train loss')
    plt.plot(range(epoch+1), dev_history_loss, 'r', label='dev loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()