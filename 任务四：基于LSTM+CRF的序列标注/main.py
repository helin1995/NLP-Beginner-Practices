# coding: utf-8
import os
import torch
import torch.nn.functional as F
import pickle as pkl
import numpy as np
from utils import *
from model import BiLSTM_CRF
from config import Config
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    torch.manual_seed(1)
    config = Config()
    datapath = config.dataPath
    trainPath = os.path.join(datapath, 'example.train')
    devPath = os.path.join(datapath, 'example.dev')
    sentences, tagSeq = loadDataSet(trainPath)
    devSent, devTags = loadDataSet(devPath)
    # for s, t in zip(sentences, tagSeq):
    #     print(s, t)
    # 训练集中实体标签及其对应的id
    tag2id = getTagSet(tagSeq)
    id2tag = {v: k for k, v in tag2id.items()}
    # print(tag2id)
    # print(id2tag)
    # 由训练集生成的词典
    if os.path.exists('./vocab/word2id.pkl'):
        with open('./vocab/word2id.pkl', 'rb') as f:
            vocab = pkl.load(f)
    else:
        vocab = getWordDict(sentences)

    config.vocab = vocab
    config.tag2id = tag2id

    log_path = config.logfile
    logger = get_logger(log_path)

    use_gpu = torch.cuda.is_available()
    model = BiLSTM_CRF(len(config.vocab), tag2id, config.embSize, config.hiddenSize)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learningRate)
    if use_gpu:
        model = model.cuda()
    num_examples = 0
    train_loss = 0.
    for epoch in range(1):
        index = np.random.permutation(range(len(sentences)))
        sentences = np.asarray(sentences)[index]
        tagSeq = np.asarray(tagSeq)[index]
        model.train()
        logger.info('Epoch: [{}/{}]'.format(epoch+1, 1))
        for sentence, tags in zip(sentences, tagSeq):
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, vocab)
            targets = torch.Tensor([tag2id[t] for t in tags]).type(torch.LongTensor)
            if use_gpu:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()
            # print(sentence_in, targets)


            loss = model.neg_log_likelihood(sentence_in, targets)
            # if use_gpu:
            #     loss = loss.cuda()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            num_examples += 1
            if num_examples % 2000 == 0:
                logger.info('train loss: {:.4f}'.format(train_loss / 2000))
                train_loss = 0.
                eval_lines = evaluate(model, config, devSent, devTags, id2tag)
                for line in eval_lines:
                    logger.info(line)