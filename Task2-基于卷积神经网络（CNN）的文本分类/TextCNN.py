# coding: utf-8

class Config(object):
    def __init__(self):
        self.maxLen = 128   #  句子的最大长度
        self.vocab = None   # 词表，在运行时赋值