import torch
import torch.nn as nn
from utils import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        '''
        :param vocab_size: 词典大小
        :param tag_to_ix: 标签词典
        :param embedding_dim: 词向量维度
        :param hidden_dim: lstm隐藏层维度
        '''
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # 标签词典大小
        # 随机生成词向量矩阵
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 双向LSTM结构，每个时刻的状态（输出）是前向和后向拼接起来的结果，大小为hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # 线性层：将双向LSTM的输出映射到标签空间中，即将hidden_dim大小的向量映射为tagset_size大小的向量
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # 标签转移矩阵
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        '''
        初始化Bi-LSTM层的h, c
        :return:
        '''
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        '''
        1、将sentence根据词向量矩阵得到这个句子的词向量表示矩阵，每行都为一个单词的词向量
        2、将句子的词向量矩阵输入给Bi-LSTM得到句子的表示
        :param sentence:
        :return:
        '''
        # 初始化Bi-LSTM层的h, c
        self.hidden = self.init_hidden()
        # 根据sentence得到它对应的词向量矩阵
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # [seq_len, 1, emb_size]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # [seq_len, hidden_dim]
        lstm_feats = self.hidden2tag(lstm_out)  # [seq_len, tag_size]
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        '''
        利用维特比算法解码得到lstm_feats对应的最大分数及其对应的路径
        :param feats: [seq_len, tag_size]
        :return:
        '''
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)  # [1, tag_size]
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path  # path_score: 最优路径对应的分数，best_path：最优标签id序列

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # 将sentence输入给双向LSTM，得到句子的表示
        lstm_feats = self._get_lstm_features(sentence)  # [seq_len, tag_size]

        # Find the best path, given the features.
        # 利用维特比算法得到lstm_feats对应的分数最大的路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq




if __name__ == '__main__':
    ### Pytorch中RNN、LSTM、GRU的使用详解参考：https://blog.csdn.net/lkangkang/article/details/89814697
    # 输入特征维度5，输出维度10, 层数1
    lstm = nn.LSTM(5, 10, 1, batch_first=True, bidirectional=True)
    # seq长度4，batch_size=2
    input = torch.randn(1, 4, 5)
    out, (hn, cn) = lstm(input)
    print(out.squeeze().size())
    print(out.squeeze())

    print(hn.size())
    print(hn)