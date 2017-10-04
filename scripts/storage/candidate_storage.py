import os
import numpy as np

from util import helper

import logging


class CandidateStorage(object):
    def __init__(self, root, n, cube_size, negatives, max_negatives, file_prefix = "", shuffle = False):
        self.root = root
        self.n = n
        self.negatives = negatives
        self.max_negatives = max_negatives
        self.index = 0
        self.neg_index = 0
        self.shuffle = shuffle
        self.file_prefix = file_prefix if file_prefix == "" else file_prefix + "_"

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        data_filename = os.path.join(self.root, "%sdata.npy" % self.file_prefix)
        self.data_shape = (n, 1, cube_size, cube_size, cube_size)
        labels_filename = os.path.join(self.root, "%slabels.npy" % self.file_prefix)

        self.reordering = np.arange(0, n, dtype = np.uint32)

        r = np.random.RandomState(42)
        if self.shuffle:
            r.shuffle(self.reordering)
        logging.debug("First ten indices, shuffling: %s" % str(self.reordering[:10]))

        take = self.max_negatives
        leave = self.negatives - self.max_negatives

        self.negative_selection = np.hstack((np.ones(take, dtype = np.uint32), np.zeros(leave, dtype = np.uint32)))
        if self.max_negatives < self.negatives:
            r.shuffle(self.negative_selection)
        logging.debug("First ten indices, negative selection: %s" % str(self.negative_selection[:10]))

        self.data = np.memmap(data_filename, dtype = helper.DTYPE, mode = "w+", shape = self.data_shape)
        self.labels = np.memmap(labels_filename, dtype = helper.DTYPE, mode = "w+", shape = n)

    def store_candidate(self, data, label):
        if label < 0.5:
            take_negative = self.negative_selection[self.neg_index]
            self.neg_index += 1
            if not take_negative:
                return 0

        store_at = self.reordering[self.index]
        self.data[store_at, :, :, :, :] = data
        self.labels[store_at] = label
        self.index += 1

        return 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.debug("Written: %d, expected: %d" % (self.index, self.n))
        assert self.index == self.n, "wrong number of candidates written"

    def store_info(self, info_object):
        info_object["type"] = type(self).__name__
        info_object["written"] = helper.now()
        info_object["revision"] = helper.git_hash()
        info_object["shape"] = self.data_shape
        info_object["sample_shape"] = self.data_shape[1:]
        info_object["samples"] = self.data_shape[0]
        info_object["shuffled"] = self.shuffle
        with open(os.path.join(self.root, "%sinfo.txt" % self.file_prefix), "w") as info_file:
            for k in info_object:
                info_file.write("%s: %s\n" % (k, info_object[k]))
