import re
import string
import random
import math
import numpy as np
import matplotlib.pyplot as plt

MAX_VOCAB_SIZE = 5000   # 词表最大长度
MIN_FREQ = 1            # 单词最小出现次数

def clean_data(train_path, stop_words):
    '''
    清洗数据：将字符串转换成小写、去掉标点符号和停用词
    :param train_path:
    :param stop_words:
    :return:
    '''
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

def tf(word, sentence):
    # 词频本来的计算方式是一个文档中该单词出现的次数除以该文档单词总数，但是在这里没有选择除以单词总数
    return sentence.count(word)

def idf(word, data):
    doc_nums = len(data)
    word_in_doc = 0
    for sentence, _ in data:
        if word in sentence:
            word_in_doc += 1
    return math.log10(doc_nums / (word_in_doc + 1))

def feature_extraction(data, vocab, n_gram):
    feature = []
    word_idf = {}
    for sentence, label in data:
        text_feature = [0] * len(vocab)
        for word in sentence:
            if word in vocab:
                text_feature[vocab[word]] = sentence.count(word)
                '''
                if word in word_idf:
                    idf_of_word = word_idf[word]
                    text_feature[vocab[word]] = tf(word, sentence) * idf_of_word
                else:
                    idf_of_word = idf(word, data)
                    text_feature[vocab[word]] = tf(word, sentence) * idf_of_word
                    word_idf[word] = idf_of_word
                '''
        feature.append([text_feature, int(label)])
    return feature


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def binary_cross_entropy_loss(config, w, y, y_hat):
    regular_term = np.sum(w ** 2)
    return -y * np.log(y_hat[0][0]) - (1 - y) * np.log(1 - y_hat[0][0]) + config.lamda / 2 * regular_term

def evaluate(config, model, dev_feature):
    dev_loss = 0.
    dev_true_nums = 0
    for x, y in dev_feature:
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y)
        y_hat = sigmoid(np.matmul(model.weight.T, x))
        y_pred = int(y_hat[0][0] > 0.5)
        dev_true_nums += int((y_pred == y))
        dev_loss += binary_cross_entropy_loss(config, model.weight, y, y_hat)
    return dev_true_nums / len(dev_feature), dev_loss / len(dev_feature)

def train(config, model, train_feature, dev_feature):
    N = len(train_feature)
    train_loss_history = []
    dev_loss_history = []
    before_dev_loss = 0.
    random.shuffle(dev_feature)
    for epoch in range(150):
        print('[{}/{}]'.format(epoch + 1, 150))
        random.shuffle(train_feature)
        total_loss = 0.
        true_nums = 0
        for x, y in train_feature:
            x = np.asarray(x).reshape(-1, 1)
            y = np.asarray(y)
            y_hat = sigmoid(np.matmul(model.weight.T, x))
            y_pred = int(y_hat[0][0] > 0.5)
            true_nums += int((y_pred == y))
            loss = binary_cross_entropy_loss(config, model.weight, y, y_hat)
            total_loss += loss
            model.weight += config.learning_rate * x * (y - y_hat[0][0]) - config.lamda * model.weight
        train_accu = true_nums / N
        train_loss = total_loss / N
        dev_accu, dev_loss = evaluate(config, model, dev_feature)
        train_loss_history.append(train_loss)
        dev_loss_history.append(dev_loss)
        print('train accuracy: {:.4f}\t train loss: {:.4f}\t dev accuracy: {:.4f}\t dev loss: {:.4f}'.format(train_accu,
                                                                                                             train_loss,
                                                                                                             dev_accu,
                                                                                                             dev_loss))
        # early stop
        if abs(dev_loss - before_dev_loss) < config.tolerance:
            break
        else:
            before_dev_loss = dev_loss
    # 绘图：train loss，dev loss
    plt.plot(range(epoch + 1), train_loss_history, 'b', label='train loss')
    plt.plot(range(epoch + 1), dev_loss_history, 'r', label='dev loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()