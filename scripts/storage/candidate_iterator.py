import argparse
import os
import random
import time
import logging
import sys

import mxnet as mx
import numpy as np

from util import helper


class SequentialIndex(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.reset()

    def reset(self):
        self.part = 0
        self.offset = 0

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, index):
        new_index = index - self.offset
        if new_index >= self.sizes[self.part]:
            self.offset += self.sizes[self.part]
            self.part += 1
            if self.part >= len(self.sizes):
                raise IndexError("SequentialIndex - index out of range")
            return self[index]
        return self.part, new_index


class RandomIndex(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.idx = np.zeros(sum(sizes), dtype = 'int32')
        offset = 0
        for i, l in enumerate(sizes):
            self.idx[offset:offset + l] = i
            offset += l
        rng = np.random.RandomState(42)
        rng.shuffle(self.idx)
        logging.debug("shuffled idx (first 20): %s" % str(self.idx[:20]))
        self.reset()

    def reset(self):
        self.cursors = [0] * len(self.sizes)

    def __len__(self):
        return sum(self.idx)

    def __getitem__(self, index):
        p = self.idx[index]
        i = self.cursors[p]
        self.cursors[p] += 1
        return p, i


class CandidateIter(mx.io.PrefetchingIter):
    def __init__(self, root, subsets, batch_size = 1, shuffle = False, chunk_size = 100, data_name = 'data', label_name = 'softmax_label'):
        self.__inner_iter = InnerIter(root, subsets, batch_size = batch_size, shuffle = shuffle,
                                      data_name = data_name, label_name = label_name)
        super(CandidateIter, self).__init__(self.__inner_iter)

    def get_info(self):
        return self.__inner_iter.get_info()

    def sizes(self):
        return self.__inner_iter.sizes()

    def total_size(self):
        return self.__inner_iter.total_size()


class InnerIter(mx.io.DataIter):
    def __init__(self, root, selection, batch_size = 1, shuffle = False, data_name = 'data', label_name = 'softmax_label'):
        self.batch_size = batch_size
        self.data_name = data_name
        self.label_name = label_name

        self.data_files = []
        self.label_files = []
        info_files_normal = {}
        info_files_tianchi = {}

        self.subsets = helper.get_filtered_subsets(root, selection)
        logging.debug("Subsets for iterator: %s" % str(self.subsets))

        for subset in self.subsets:
            info_file_data = helper.read_info_file(os.path.join(root, subset, "info.txt"))
            shape = info_file_data["shape"]
            if len(shape) == 4:
                shape = shape[:1] + [1] + shape[1:]

            data_path = os.path.join(root, subset, "data.npy")
            self.data_files.append(np.memmap(data_path, dtype = helper.DTYPE, mode = "r"))
            self.data_files[-1].shape = shape

            label_path = os.path.join(root, subset, "labels.npy")
            self.label_files.append(np.memmap(label_path, dtype = helper.DTYPE, mode = "r"))
            self.label_files[-1].shape = (shape[0],)

            if not helper.is_tianchi_dataset(root) and int(subset.replace("subset", "")) >= 10:
                info_files_tianchi[subset] = info_file_data
            else:
                info_files_normal[subset] = info_file_data

        self.batch_shape = tuple([self.batch_size] + [1] + info_file_data["shape"][2:])

        if len(info_files_normal) == 0:
            info_files_normal = info_files_tianchi
        self.info = helper.check_and_combine(info_files_normal)
        logging.debug("Info %s: %s" % (root, self.info))

        self.needs_shuffling = shuffle and self.info["shuffled"] == "False"
        logging.debug("Needs shuffling: %s" % self.needs_shuffling)

        if self.needs_shuffling:
            logging.warning("The dataset was not shuffled while preparation so reading the shuffled version will be extremely slow!")
            self.idx = RandomIndex([len(labels) for labels in self.label_files])
        else:
            self.idx = SequentialIndex([len(labels) for labels in self.label_files])

        self.reset()

    def get_info(self):
        return self.info

    def sizes(self):
        sizes = {}
        for i, subset in enumerate(self.subsets):
            sizes[subset] = len(self.label_files[i])
        return sizes

    def total_size(self):
        sizes = self.sizes()
        return sum([sizes[x] for x in sizes])

    def __iter__(self):
        return self

    def reset(self):
        self.cursor = 0
        self.idx.reset()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        data_description = mx.io.DataDesc(self.data_name, self.batch_shape, "f4")
        data_description.layout = "NCDHW"
        return [data_description]

    @property
    def provide_label(self):
        label_description = mx.io.DataDesc(self.label_name, (self.batch_size, ), "f4")
        return [label_description]

    def next(self):
        current_batch_size = 0
        data = np.zeros(self.batch_shape, dtype=helper.DTYPE)
        labels = np.zeros(self.batch_size, dtype=helper.DTYPE)
        while current_batch_size < self.batch_size:
            if self.cursor >= len(self.idx):
                if current_batch_size == 0:
                    raise StopIteration
                else:
                    break

            p, i = self.idx[self.cursor]

            data[current_batch_size] = self.data_files[p][i]
            labels[current_batch_size] = self.label_files[p][i]
            self.cursor += 1
            current_batch_size += 1

        pad = self.batch_size - current_batch_size
        return mx.io.DataBatch(data=[mx.io.array(data, dtype="f4")], label=[mx.io.array(labels, dtype="f4")], pad=pad, index=None)
