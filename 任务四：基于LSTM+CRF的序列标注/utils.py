# coding: utf-8

import os
import torch
import logging
import pickle as pkl
import numpy as np

MIN_FREQ = 1
MAX_VOCAB_SIZE = 10000
MAX_LEN = 32
UNK = '<unk>'
PAD = '<pad>'
START_TAG = "<START>"
STOP_TAG = "<STOP>"

def loadDataSet(datapath):
    with open(datapath, 'r', encoding='utf-8') as f:
        sentences, tagSeq = [], []
        sentenceNum = 0
        s, t = [], []
        lineNum = 1
        for line in f.readlines():
            print('第' + str(lineNum) + '行：' + line.strip())
            line = line.strip()
            if line == '':
                if s == []:
                    lineNum += 1
                    continue
                else:
                    lineNum += 1
                    sentences.append(s)
                    tagSeq.append(t)
                    s, t = [], []
                    sentenceNum += 1
            else:
                # word, _, _, tag = line.split()
                word, tag = line.split()
                s.append(word)
                t.append(tag)
                lineNum += 1
        sentences.append(s)
        tagSeq.append(t)
        sentenceNum += 1
        print('总共有' + str(len(tagSeq)) + '句子')
        return sentences, tagSeq

def getTagSet(tagSeq):
    tag2id = {}
    tagList = []
    for tag in tagSeq:
        for t in tag:
            if t not in tagList:
                tagList.append(t)
    for idx, tag in enumerate(tagList):
        tag2id[tag] = idx
    tag2id[START_TAG] = len(tag2id)
    tag2id[STOP_TAG] = len(tag2id)
    return tag2id

def getWordDict(sentences):
    vocab = dict()
    for word_list in sentences:
        for word in word_list:
            vocab[word] = vocab.get(word, 0) + 1  # 统计词频
    vocab_list = sorted([_ for _ in vocab.items() if _[1] > MIN_FREQ], key=lambda x: x[1], reverse=True)[
                 :MAX_VOCAB_SIZE]
    vocab = {_[0]: idx for idx, _ in enumerate(vocab_list)}
    vocab[UNK], vocab[PAD] = len(vocab), len(vocab) + 1
    with open('./vocab/word2id.pkl', 'wb') as f:
        pkl.dump(vocab, f, protocol=pkl.HIGHEST_PROTOCOL)
    return vocab

def seq2id(sentences, tagSeq, vocab, tag2id):
    dataSet, tagSet = [], []
    for sent, tags in zip(sentences, tagSeq):
        if len(sent) <= MAX_LEN:
            tags += ['O'] * (MAX_LEN - len(sent))
            sent += [PAD] * (MAX_LEN - len(sent))
        else:
            sent = sent[:MAX_LEN]
            tags = tags[:MAX_LEN]
        sid, tid = [], []
        for word, tag in zip(sent, tags):
            sid.append(vocab.get(word, vocab[UNK]))
            tid.append(tag2id.get(tag))
        dataSet.append(sid)
        tagSet.append(tid)
    return torch.from_numpy(np.asarray(dataSet)), torch.from_numpy(np.asarray(tagSet))

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    '''将单词序列转换为对应id的序列
    :param seq: 单词列表['i', 'love', 'china']
    :param to_ix: 词典
    :return: [0, 1, 2]
    '''
    # print(seq, to_ix)
    idxs = [to_ix.get(w, to_ix[UNK]) for w in seq]
    return torch.Tensor(idxs).type(torch.LongTensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def evaluate(model, config, devSent, devTags, id2tag):
    model.eval()
    script_file = "conlleval"
    output_file = os.path.join(config.result_file, "ner_predict.utf8")
    result_file = os.path.join(config.result_file, "ner_result.utf8")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, gold_tags in zip(devSent, devTags):
            sentence_in = prepare_sequence(sentence, config.vocab)
            if torch.cuda.is_available():
                sentence_in = sentence_in.cuda()
            _, tag_seq = model(sentence_in)
            pred_tags = [id2tag[t] for t in tag_seq]
            for word, gold, pred in zip(sentence, gold_tags, pred_tags):
                f.write(word + ' ' + gold + ' ' + pred)
                f.write('\n')
            f.write('\n')
    os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
    eval_lines = []
    with open(result_file) as f:
        for line in f:
            eval_lines.append(line.strip())
    return eval_lines