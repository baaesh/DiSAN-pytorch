from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize
import numpy as np


class SNLI():
	def __init__(self, args):
		self.TEXT = data.Field(batch_first=True, include_lengths=True, tokenize=word_tokenize, lower=True)
		self.LABEL = data.Field(sequential=False, unk_token=None)

		self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

		for e in self.train.examples:
			if len(e.premise) > 25 or len(e.hypothesis) > 25:
				self.train.examples.remove(e)
		for e in self.dev.examples:
			if len(e.premise) > 25 or len(e.hypothesis) > 25:
				self.dev.examples.remove(e)
		for e in self.test.examples:
			if len(e.premise) > 25 or len(e.hypothesis) > 25:
				self.test.examples.remove(e)

		self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='6B', dim=300))
		self.LABEL.build_vocab(self.train)

		self.train_iter, self.dev_iter, self.test_iter = \
			data.BucketIterator.splits((self.train, self.dev, self.test),
									   batch_size=args.batch_size,
									   device=args.gpu)

