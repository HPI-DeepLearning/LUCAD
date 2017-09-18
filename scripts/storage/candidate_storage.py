import os

import numpy as np

from util import helper


class CandidateStorage(object):
    def __init__(self, root, n, cube_size, file_prefix = "", shuffle = False):
        self.root = root
        self.n = n
        self.index = 0
        self.shuffle = shuffle
        self.file_prefix = file_prefix if file_prefix == "" else file_prefix + "_"

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        data_filename = os.path.join(self.root, "%sdata.npy" % self.file_prefix)
        self.data_shape = (n, 1, cube_size, cube_size, cube_size)
        labels_filename = os.path.join(self.root, "%slabels.npy" % self.file_prefix)

        self.data = np.memmap(data_filename, dtype = helper.DTYPE, mode = "w+", shape = self.data_shape)
        self.labels = np.memmap(labels_filename, dtype = helper.DTYPE, mode = "w+", shape = n)

    def store_candidate(self, data, label):
        self.data[self.index, :, :, :, :] = data
        self.labels[self.index] = label
        self.index += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.shuffle:
            r = np.random.RandomState(i)
            r.shuffle(self.data)
            r = np.random.RandomState(i)
            r.shuffle(self.labels)

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
