# coding: utf-8

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from TextCNN import Config, Model
from sklearn import metrics

if __name__ == '__main__':
    token_pattern = r'\b[A-Za-z][A-Za-z]+\b'  # 正则表达式：只取每篇文档中两端是字母的词
    trainData = load_data('./dataset/train.csv', token_pattern)
    vocab = build_vocab(trainData)

    config = Config()
    config.vocab = vocab

    trainX, trainY = generate_data(trainData, vocab, config)
    train = TensorDataset(trainX, trainY)
    loader = DataLoader(dataset=train, batch_size=config.batchSize, shuffle=True, num_workers=0)

    model = Model(config)
    # criterion = F.cross_entropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learningRate)

    for epoch in range(1):
        print('Epoch: [{}/{}]'.format(epoch + 1, 10))
        train_loss = 0.
        for x, y in loader:
            y_ = model(x)
            loss = F.cross_entropy(input=y_, target=y)
            y_pred = torch.argmax(y_, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item() * x.size()[0]
        print(train_loss / 2400)