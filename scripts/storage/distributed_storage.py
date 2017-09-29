import os
import random

import numpy as np

from util import helper


def padded_format(i, maximum):
    return ("%0" + str(len(str(maximum))) + "d") % i


class DistributedStorage(object):
    def __init__(self, root, n, cube_size, neg, neg_d, shuffle = True, parts = 10):
        self.root = root
        self.n = n
        self.index = 0
        self.data_shape = [1, cube_size, cube_size, cube_size]
        self.shuffle = shuffle
        self.parts = parts
        self.part_target = range(0, parts)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if self.shuffle:
            random.seed(self.n)

        self.data_maps = []
        self.label_maps = []
        s = 0
        for i in range(0, parts):
            s += self.get_num_elements(i)
            df = self.get_data_filename(i, True)
            lf = self.get_labels_filename(i, True)

            self.data_maps.append(np.memmap(df, dtype = helper.DTYPE, mode = "w+", shape = self.get_data_shape(i)))
            self.label_maps.append(np.memmap(lf, dtype = helper.DTYPE, mode = "w+", shape = self.get_label_shape(i)))

        assert s == self.n, "Something went wrong when calculating sizes of parts!"

    def get_num_elements(self, part):
        n0, r = divmod(self.n, self.parts)
        if self.shuffle:
            return n0 + (1 if part < r else 0)
        else:
            return n0 + (0 if (part < self.parts - 1) else r)

    def get_data_shape(self, part):
        return tuple([self.get_num_elements(part)] + self.data_shape)

    def get_label_shape(self, part):
        return self.get_num_elements(part)

    def get_data_filename(self, i, temp = False):
        format_ = "tmp_data_%s.npy" if temp else "data_%s.npy"
        return os.path.join(self.root, format_ % padded_format(i, self.parts))

    def get_labels_filename(self, i, temp = False):
        format_ = "tmp_labels_%s.npy" if temp else "labels_%s.npy"
        return os.path.join(self.root, format_ % padded_format(i, self.parts))

    def store_candidate(self, data, label):
        if self.shuffle:
            i0, p = divmod(self.index, self.parts)
            n0, r = divmod(self.n, self.parts)
            if p == 0:
                random.shuffle(self.part_target)
            if self.index < self.n - r:
                p = self.part_target[p]
        else:
            p = 0
            i0 = self.index
            while i0 >= self.get_num_elements(p):
                i0 -= self.get_num_elements(p)
                p += 1
        self.data_maps[p][i0] = data
        self.label_maps[p][i0] = label
        self.index += 1

    def store_info(self, info_object):
        info_object["type"] = type(self).__name__
        info_object["written"] = helper.now()
        info_object["revision"] = helper.git_hash()
        info_object["shape"] = tuple([self.n] + self.data_shape)
        info_object["sample_shape"] = self.data_shape
        info_object["samples"] = self.n
        info_object["shuffled"] = self.shuffle
        with open(os.path.join(self.root, "info.txt"), "w") as info_file:
            for k in info_object:
                info_file.write("%s: %s\n" % (k, info_object[k]))

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        # rewrite all files with shape information
        for i in range(0, self.parts):
            data = np.copy(self.data_maps[i])
            labels = np.copy(self.label_maps[i])

            self.data_maps[i] = None
            self.label_maps[i] = None

            if self.shuffle:
                r = np.random.RandomState(i)
                r.shuffle(data)
                r = np.random.RandomState(i)
                r.shuffle(labels)

            np.save(self.get_data_filename(i), data)
            np.save(self.get_labels_filename(i), labels)

            # remove initial memmaps
            os.remove(self.get_data_filename(i, True))
            os.remove(self.get_labels_filename(i, True))
