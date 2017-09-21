import argparse
import os
import random
import time

import mxnet as mx
import numpy as np

from util import helper


class CandidateIter(mx.io.DataIter):
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
        self.reset()

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

            if end - start <= self.batch_size:
                self.current_chunk = 0
                self.current_file += 1
                continue

            self.__iterator = mx.io.NDArrayIter(data = data[start:end, :, :, :, :], label = labels[start:end],
                                                batch_size = self.batch_size, shuffle = self.shuffle,
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


def main(args):
    start = time.time()
    chunk_start = start

    data_iter = CandidateIter(args.root, args.subsets, batch_size = 30, shuffle = True)

    print "Sizes: %s" % data_iter.sizes()
    print "Total number of samples: %d" % data_iter.total_size()
    print "Data layout: %s" % data_iter.provide_data[0].layout
    print "Data shape: %s" % str(data_iter.provide_data[0].shape)

    i = 0
    for batch in data_iter:
        if i == 0:
            print "Batch shape: %s" % ",".join([str(x) for x in batch.data[0].shape])
        if i % 100 == 99:
            prev_chunk = chunk_start
            chunk_start = time.time()
            chunk = chunk_start - prev_chunk
            avg = (chunk_start - start) / ((i+1)/100.0)
            print "Batch %d (%.2f, total avg: %.2f sec / 100 batch)!" % (i + 1, chunk, avg)
        assert len(batch.data[0]) == 30
        assert len(batch.label[0]) == 30
        i += 1
    print "Finished!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test the dataset iterator")
    parser.add_argument("root", type=str, help="folder containing prepared dataset folders")
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = range(0, 10))
    main(parser.parse_args())

