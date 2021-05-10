import re
import string
import random
import numpy as np
import matplotlib.pyplot as plt

MAX_VOCAB_SIZE = 5000   # 词表最大长度
MIN_FREQ = 1            # 单词最小出现次数

def clean_data(train_path, stop_words):
    # pattern = re.compile(r'\b[A-Za-z][A-Za-z]+\b')
    data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().lower()
            content = line[1:-3]
            label = line[-1]
            sentence = [word for word in content.split(' ') if word not in stop_words and word not in string.punctuation]
            # word_list = pattern.findall(content)
            # sentence = [word for word in word_list if word not in stop_words]
            data.append([sentence, label])
    return data

def build_vocab(data, n_gram):
    vocab = dict()
    for sentence, _ in data:
        for i in range(len(sentence)):
            if i+n_gram <= len(sentence):
                word_group = ' '.join(sentence[i:i+n_gram])
                vocab[word_group] = vocab.get(word_group, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] > MIN_FREQ], key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE]
    vocab = {_[0]: idx for idx, _ in enumerate(vocab_list)}
    return vocab

def feature_extraction(data, vocab):
    feature = []
    for sentence, label in data:
        text_feature = [0] * len(vocab)
        for word in sentence:
            if word in vocab:
                text_feature[vocab[word]] = sentence.count(word)
        feature.append([text_feature, int(label)])
    return feature


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def binary_cross_entropy_loss(w, y, y_hat):
    lamda = 1e-4
    regular = np.sum(w ** 2)
    return -y * np.log(y_hat[0][0]) - (1 - y) * np.log(1 - y_hat[0][0]) + lamda / 2 * regular

def train(config, model, train_feature, dev_feature):
    N = len(train_feature)
    M = len(dev_feature)
    train_loss_history = []
    dev_loss_history = []
    before_dev_loss = 0.
    for epoch in range(100):
        print('[{}/{}]'.format(epoch + 1, 100))
        random.shuffle(train_feature)
        total_loss = 0.
        true_nums = 0
        dev_loss = 0.
        dev_true_nums = 0
        for x, y in train_feature:
            x = np.asarray(x).reshape(-1, 1)
            y = np.asarray(y)
            y_hat = sigmoid(np.matmul(model.weight.T, x))
            y_pred = int(y_hat[0][0] > 0.5)
            true_nums += int((y_pred == y))
            loss = binary_cross_entropy_loss(model.weight, y, y_hat)
            total_loss += loss
            model.weight += config.learning_rate * x * (y - y_hat[0][0]) - config.lamda * model.weight
        for x, y in dev_feature:
            x = np.asarray(x).reshape(-1, 1)
            y = np.asarray(y)
            y_hat = sigmoid(np.matmul(model.weight.T, x))
            y_pred = int(y_hat[0][0] > 0.5)
            dev_true_nums += int((y_pred == y))
            dev_loss += binary_cross_entropy_loss(model.weight, y, y_hat)
        train_loss_history.append(total_loss / N)
        dev_loss_history.append(dev_loss / M)
        print('train accuracy: {:.4f}\t train loss: {:.4f}\t dev accuracy: {:.4f}\t dev loss: {:.4f}'.format(
            true_nums / N,
            total_loss / N,
            dev_true_nums / M,
            dev_loss / M))
        if abs(dev_loss / M - before_dev_loss) < config.tolerance:
            break
        else:
            before_dev_loss = dev_loss / M

    plt.plot(range(epoch + 1), train_loss_history, 'b')
    plt.plot(range(epoch + 1), dev_loss_history, 'r')
    plt.show()