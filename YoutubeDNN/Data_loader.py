import os
from os import listdir
from os.path import isfile, join
import numpy as np
import time


# This Data_Loader file is copied online

def INFO_LOG(info):
	print("[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info))


class Data_Loader:
	def __init__(self, options):
		self.pad = "<PAD>"
		self.positive_data_file = options['dir_name']
		positive_examples = list(open(self.positive_data_file, "r").readlines())
		positive_examples = [s.strip().split(",") for s in positive_examples]
		self.max_len = options['max_len']

		train_set, valid_set, test_set = self.split_data(positive_examples)
		train_set = self.cut_filter_seq(train_set)

		self.item_fre = {}
		self.build_map(train_set)
		self.build_map(valid_set)
		self.build_map(test_set)
		self.items_voc = self.item_fre.keys()
		self.item2id = dict(zip(self.items_voc, range(1, len(self.items_voc) + 1)))
		self.padid = 0
		self.id2item = {value: key for key, value in self.item2id.items()}
		INFO_LOG("Vocab size:{} + 1 (pad)".format(self.size()))

		self.train_set, self.valid_set, self.test_set = np.array(self.getSamplesid(train_set)), \
		                                                np.array(self.getSamplesid(valid_set)), \
		                                                np.array(self.getSamplesid(test_set))

	def split_data(self, user_seq):
		train_set, valid_set, test_set = [], [], []
		if 'retail' in self.positive_data_file:
			print('split retail dataset')
			for seq in user_seq:
				if len(seq) >= 5 and len(set(seq)) > 3:
					test_set.append(seq[max(len(seq) - self.max_len, 0):])
					valid_set.append(seq[max(len(seq) - 1 - self.max_len, 0):-1])
					train_set.append(seq[:-2])
		else:
			for seq in user_seq:
				if len(seq) >= 5:
					test_set.append(seq[max(len(seq) - self.max_len, 0):])
					valid_set.append(seq[max(len(seq) - 1 - self.max_len, 0):-1])
					train_set.append(seq[:-2])
		return train_set, valid_set, test_set

	def cut_filter_seq(self, data):
		samples = []
		for seq in data:
			if len(seq) > self.max_len:
				idx = 0
				start = len(seq) % self.max_len
				if start > 1:  # filter len(seq) < 2
					samples.append(seq[0:start])
				while (start + (idx + 1) * self.max_len) <= len(seq):
					temp = seq[start + idx * self.max_len: start + (idx + 1) * self.max_len]
					idx += 1
					samples.append(temp)
			else:
				samples.append(seq)
		return samples

	def build_map(self, data):
		for sample in data:
			for item in sample:
				if item in self.item_fre.keys():
					self.item_fre[item] += 1
				else:
					self.item_fre[item] = 1

	def sample2id(self, sample):
		sample2id = []
		for s in sample:
			sample2id.append(self.item2id[s])
		sample2id = [self.padid] * (self.max_len - len(sample2id)) + sample2id
		return sample2id

	def getSamplesid(self, samples):
		samples2id = []
		for sample in samples:
			samples2id.append(self.sample2id(sample))

		return samples2id

	def size(self):
		return len(self.item2id)

	def load_generator_data(self, sample_size):
		text = self.text
		mod_size = len(text) - len(text) % sample_size
		text = text[0:mod_size]
		text = text.reshape(-1, sample_size)
		return text, self.vocab_indexed

	def string_to_indices(self, sentence, vocab):
		indices = [self.item2id[s] for s in sentence.split(',')]
		return indices

	def inidices_to_string(self, sentence, vocab):
		id_ch = {vocab[ch]: ch for ch in vocab}
		sent = []
		for c in sentence:
			if id_ch[c] == 'eol':
				break
			sent += id_ch[c]

		return "".join(sent)


