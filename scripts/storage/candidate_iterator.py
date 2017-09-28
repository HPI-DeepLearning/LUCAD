import argparse
import os
import random
import time
import logging
import sys

import mxnet as mx
import numpy as np

from util import helper


class CandidateIter(mx.io.PrefetchingIter):
    def __init__(self, root, subsets, batch_size = 1, shuffle = False, chunk_size = 100, data_name = 'data', label_name = 'softmax_label'):
        self.__inner_iter = InnerIter(root, subsets, batch_size = batch_size, shuffle = shuffle,
                                      chunk_size = 1, data_name = data_name, label_name = label_name)
        super(CandidateIter, self).__init__(self.__inner_iter)

    def get_info(self):
        return self.__inner_iter.get_info()

    def sizes(self):
        return self.__inner_iter.sizes()

    def total_size(self):
        return self.__inner_iter.total_size()


class InnerIter(mx.io.DataIter):
    def __init__(self, root, subsets, batch_size = 1, shuffle = False, chunk_size = 100, data_name = 'data', label_name = 'softmax_label'):
        self.data_files = []
        self.label_files = []
        self.info_files = []
        for subset in subsets:
            self.data_files.append(os.path.join(root, "subset%d" % subset, "data.npy"))
            self.label_files.append(os.path.join(root, "subset%d" % subset, "labels.npy"))
            self.info_files.append(os.path.join(root, "subset%d" % subset, "info.txt"))

        info_files = {}
        for subset in subsets:
            info_files[subset] = helper.read_info_file(os.path.join(root, "subset%d" % subset, "info.txt"))

        self.info = helper.check_and_combine(info_files)
        logging.debug("Info %s: %s" % (root, self.info))

        if shuffle:
            random.seed(42)
            random.shuffle(self.data_files)
            random.seed(42)
            random.shuffle(self.label_files)
            random.seed(42)
            random.shuffle(self.info_files)

        self.shuffle = shuffle
        self.root = root
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.data_name = data_name
        self.label_name = label_name

        self.needs_shuffling = self.shuffle and self.info["shuffled"] == "False"
        logging.debug("Needs shuffling: %s" % self.needs_shuffling)

        self.current_file = 0
        self.current_chunk = 0
        self.__iterator = None

    def get_info(self):
        return self.info

    def sizes(self):
        sizes = {}
        for filename in self.label_files:
            labels = np.memmap(filename, dtype = helper.DTYPE, mode = "r")
            sizes[filename] = len(labels)
        return sizes

    def total_size(self):
        sizes = self.sizes()
        return sum([sizes[x] for x in sizes])

    def get_current_iterator(self):
        while self.__iterator is None:
            if self.current_file >= len(self.data_files):
                raise StopIteration

            info = helper.read_info_file(self.info_files[self.current_file])
            shape = info["shape"]

            if len(shape) == 4:
                shape = shape[:1] + [1] + shape[1:]

            data = np.memmap(self.data_files[self.current_file], dtype = helper.DTYPE, mode = "r")
            data.shape = shape

            labels = np.memmap(self.label_files[self.current_file], dtype = helper.DTYPE, mode = "r")
            labels.shape = (shape[0])

            start = self.current_chunk * self.chunk_size * self.batch_size
            end = min((self.current_chunk + 1) * self.chunk_size * self.batch_size, data.shape[0])

            self.current_chunk += 1

            if end - start < self.batch_size:
                self.current_chunk = 0
                self.current_file += 1
                continue

            # logging.debug("Create new NDArrayIter, %s - %d:%d" % (self.data_files[self.current_file], start, end))
            self.__iterator = mx.io.NDArrayIter(data = data[start:end, :, :, :, :], label = labels[start:end],
                                                batch_size = self.batch_size, shuffle = self.needs_shuffling,
                                                data_name = self.data_name, label_name = self.label_name)

        return self.__iterator

    def __iter__(self):
        return self

    def reset(self):
        self.current_file = 0
        self.current_chunk = 0
        self.__iterator = None

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        data = self.get_current_iterator().provide_data
        data[0].layout = "NCDHW"
        return data

    @property
    def provide_label(self):
        return self.get_current_iterator().provide_label

    def next(self):
        result = None
        while result is None:
            iterator = self.get_current_iterator()
            try:
                result = iterator.next()
            except StopIteration:
                self.__iterator = None
        return result
