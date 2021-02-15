import os
from os import listdir
from os.path import isfile, join
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from collections import OrderedDict

# This Data_Loader file is copied online

def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))


class Data_loader:
    def __init__(self, options):
        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip().split(",") for s in positive_examples]
        train_seq, valid_seq, test_seq = self.split_data(positive_examples)

        self.item_map = {}
        self.build_map(train_seq)
        self.build_map(valid_seq)
        self.build_map(test_seq)
        self.items_voc = self.item_map.keys()
        self.item2id = dict(zip(self.items_voc, range(len(self.items_voc))))
        self.train_seq, self.valid_seq, self.test_seq = self.getSamplesid(train_seq), \
                                         self.getSamplesid(valid_seq), \
                                         self.getSamplesid(test_seq)

        self.train_set, self.valid_set, self.test_set = self.getPairSmaple(self.train_seq),\
                                                        self.getPairSmaple(self.valid_seq), \
                                                        self.getPairSmaple(self.test_seq)
        self.user_size = len(test_seq)
        self.item_size = len(self.items_voc)

    def split_data(self, user_seq):
        train_set, valid_set, test_set = [], [], []
        for seq in user_seq:
            if len(seq) >= 5 and len(set(seq)) > 3:
                ord_seq = OrderedDict().fromkeys(reversed(seq)).keys()
                ord_seq = list(reversed(ord_seq))

                test_set.append(ord_seq[-1:])
                valid_set.append(ord_seq[-2:-1])
                train_set.append(ord_seq[:-2])
        return train_set, valid_set, test_set

    def getPairSmaple(self, data):
        samples = []
        for idx, seq in enumerate(data):
            samples.extend([[idx, s] for s in seq])
        # print(samples)
        return np.array(samples)

    def build_map(self, data):
        for sample in data:
            for item in sample:
                if item in self.item_map.keys():
                    self.item_map[item] += 1
                else:
                    self.item_map[item] = 1

    def sample2id(self, sample):
        sample2id = []
        for s in sample:
            sample2id.append(self.item2id[s])
        return sample2id

    def getSamplesid(self, samples):
        samples2id = []
        for sample in samples:
            samples2id.append(self.sample2id(sample))
        return samples2id
