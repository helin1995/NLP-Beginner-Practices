import numpy as np

class Config(object):
    def __init__(self):
        self.vocab = {}
        self.n_gram = 1
        self.learning_rate = 1e-4
        self.lamda = 1e-5
        self.tolerance = 1e-3


class Model(object):
    def __init__(self, config):
        self.weight = np.zeros((len(config.vocab), 1))