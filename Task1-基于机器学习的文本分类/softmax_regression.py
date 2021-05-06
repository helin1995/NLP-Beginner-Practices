# coding: utf-8

import numpy as np

class Model(object):
    def __init__(self):
        self.W = np.random.randn(5, 10000)
        self.b = np.random.randn(5, 1)
        self.Weight = np.concatenate((self.W, self.b), axis=1)

    def softmax(self, x):
        # x: [5, batch_size]
        ex = np.exp(x)  # [5, batch_size]
        row, col = ex.shape
        ex_sum = np.sum(ex, axis=0)  # [1, batch_size]
        for i in range(col):
            for j in range(row):
                ex[j][i] /= ex_sum[i]
        return ex

    def forward(self, x):
        res = np.matmul(self.Weight, x.T)
        res = self.softmax(res)
        return res
