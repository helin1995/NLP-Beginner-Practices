import torch.cuda
import os

class Config():
    def __init__(self):
        self.learningRate = 0.01
        self.embSize = 100
        self.hiddenSize = 100
        self.vocab = None
        self.tag2id = None
        self.logfile = os.path.join('log', 'train.log')
        self.dataPath = './dataset/data/'
        self.result_file = './result'