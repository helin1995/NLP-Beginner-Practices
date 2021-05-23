# coding: utf-8

import re
import csv

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
            vocab[word] = vocab.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] > MIN_FREQ], key=lambda x: x[1], reverse=True)[
                 :MAX_VOCAB_SIZE]
    vocab = {_[0]: idx for idx, _ in enumerate(vocab_list)}
    vocab[UNK], vocab[PAD] = len(vocab), len(vocab) + 1
    return vocab

def generate_data(data, vocab, config):
    data_set = []
    for words, target in data:
        if len(words) <= config.maxLen:
            words += [PAD] * (config.maxLen - len(words))
        else:
            words = words[:config.maxLen]
        word_index = []
        for word in words:
            word_index.append(vocab.get(word, vocab[UNK]))
        data_set.append([word_index, int(target)-1])
    return data_set