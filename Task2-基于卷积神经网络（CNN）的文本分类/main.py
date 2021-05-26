# coding: utf-8

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from TextCNN import Config, Model
from sklearn import metrics

if __name__ == '__main__':
    token_pattern = r'\b[A-Za-z][A-Za-z]+\b'  # 正则表达式：只取每篇文档中两端是字母的词
    # 加载训练数据、开发数据和测试数据
    trainData = load_data('./dataset/train.csv', token_pattern)
    devData = load_data('./dataset/dev.csv', token_pattern)
    testData = load_data('./dataset/test.csv', token_pattern)
    # 根据训练集构建词典
    vocab = build_vocab(trainData)
    # 模型参数配置
    config = Config()
    # 更新模型参数配置中的词典
    config.vocab = vocab
    # 生成训练数据的加载器，并对训练集打乱顺序，批处理
    trainX, trainY = generate_data(trainData, vocab, config)
    train = TensorDataset(trainX, trainY)
    train_loader = DataLoader(dataset=train, batch_size=config.batchSize, shuffle=True, num_workers=0)
    # 开发集加载器
    devX, devY = generate_data(devData, vocab, config)
    dev = TensorDataset(devX, devY)
    dev_loader = DataLoader(dataset=dev, batch_size=config.batchSize, shuffle=False, num_workers=0)
    # 测试集加载器
    testX, testY = generate_data(testData, vocab, config)
    test = TensorDataset(testX, testY)
    test_loader = DataLoader(dataset=test, batch_size=config.batchSize, shuffle=False, num_workers=0)
    # 构建模型
    model = Model(config)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate)
    # 上一次在开发集上的loss，设为无穷大
    last_dev_loss = float('inf')
    # 开始训练
    for epoch in range(50):
        print('Epoch: [{}/{}]'.format(epoch + 1, 50))
        train_loss = 0.
        train_predict_label = np.array([])
        train_true_label = np.array([])
        model.train()
        for x, y in train_loader:
            y_ = model(x)
            loss = F.cross_entropy(input=y_, target=y)
            y_pred = torch.argmax(y_, dim=1).detach().numpy()
            train_predict_label = np.append(train_predict_label, y_pred)
            train_true_label = np.append(train_true_label, y.detach().numpy())
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
        # 在开发集上进行评价
        dev_acc, dev_loss = evaluate(model, dev_loader)

        train_acc = metrics.accuracy_score(train_true_label, train_predict_label)
        train_loss = train_loss / len(train_loader)
        if dev_loss < last_dev_loss:
            improve = '*'
            # 保存模型
            torch.save(model, config.save_path)
            last_dev_loss = dev_loss
            improve_epoch = epoch
        else:
            improve = ''
        print_msg = 'train loss: {:.4f} \t train accuracy: {:.4f} \t dev loss: {:.4f} \t dev accuracy: {:.4f} \t {}'
        print(print_msg.format(train_loss, train_acc, dev_loss, dev_acc, improve))
        # early stopping
        if epoch - improve_epoch >= 2:
            break
    # 在测试集上进行测试
    model = torch.load(config.save_path)
    model.eval()
    test_loss = 0.
    test_predict_label = np.array([])
    test_true_label = np.array([])
    for x, y in test_loader:
        y_ = model(x)
        loss = F.cross_entropy(input=y_, target=y)
        y_pred = torch.argmax(y_, dim=1).detach().numpy()
        test_predict_label = np.append(test_predict_label, y_pred)
        test_true_label = np.append(test_true_label, y.detach().numpy())

        test_loss += loss.detach().item()

    test_acc = metrics.accuracy_score(test_true_label, test_predict_label)
    test_loss = test_loss / len(test_loader)
    print('test accuracy: {:.4f} \t test loss: {:.4f}'.format(test_acc, test_loss))