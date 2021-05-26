# coding: utf-8

import re
import csv
import torch
import pickle as pkl
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from TextCNN import Config


MAX_VOCAB_SIZE = 5000   # 词表最大长度
MIN_FREQ = 1            # 单词最小出现次数
UNK = '<UNK>'  # 未知字符
PAD = '<PAD>'  # 填充字符

def load_data(path, pattern):
    '''
    加载数据并对数据进行处理
    :param path: 数据文件路径
    :param pattern: 清洗数据的正则表达式
    :return:
    '''
    p = re.compile(pattern)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label, content = row
            word_list = p.findall(content.strip().lower())
            data.append([word_list, label])
    return data

def build_vocab(data):
    vocab = dict()
    for word_list, _ in data:
        for word in word_list:
            vocab[word] = vocab.get(word, 0) + 1  # 统计词频
    vocab_list = sorted([_ for _ in vocab.items() if _[1] > MIN_FREQ], key=lambda x: x[1], reverse=True)[
                 :MAX_VOCAB_SIZE]
    vocab = {_[0]: idx for idx, _ in enumerate(vocab_list)}
    vocab[UNK], vocab[PAD] = len(vocab), len(vocab) + 1
    with open('./vocab/word2id.pkl', 'wb') as f:
        pkl.dump(vocab, f, protocol=pkl.HIGHEST_PROTOCOL)
    return vocab

def generate_data(data, vocab, config):
    data_set = []
    label = []
    for words, target in data:
        label.append(int(target)-1)
        if len(words) <= config.maxLen:
            words += [PAD] * (config.maxLen - len(words))
        else:
            words = words[:config.maxLen]
        word_index = []
        for word in words:
            word_index.append(vocab.get(word, vocab[UNK]))
        data_set.append(word_index)
    return torch.from_numpy(np.asarray(data_set)), torch.from_numpy(np.asarray(label))

def evaluate(model, dev_loader):
    model.eval()
    dev_loss = 0.
    dev_predict_label = np.array([])
    dev_true_label = np.array([])
    for x, y in dev_loader:
        y_ = model(x)
        loss = F.cross_entropy(input=y_, target=y)
        y_pred = torch.argmax(y_, dim=1).detach().numpy()
        dev_predict_label = np.append(dev_predict_label, y_pred)
        dev_true_label = np.append(dev_true_label, y.detach().numpy())

        dev_loss += loss.detach().item()

    dev_acc = metrics.accuracy_score(dev_true_label, dev_predict_label)
    dev_loss = dev_loss / len(dev_loader)
    return dev_acc, dev_loss

if __name__ == '__main__':
    # 加载预训练词向量
    config = Config()
    glove_path = './glove.6B/glove.6B.100d.txt'
    with open('./vocab/word2id.pkl', 'rb') as f:
        word2id = pkl.load(f)
    np.random.seed(123)
    embedding = np.random.randn(len(word2id), config.embedding_dim)
    glove = dict()
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            word = line[0]
            vector = np.array([float(val) for val in line[1:]]).astype('float32')
            if word in word2id.keys():
                embedding[word2id[word]] = vector
    np.save('./pretrained_wordvector/pretrained_embedding', embedding)
