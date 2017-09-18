import argparse
import os
import random
import time
import threading

import mxnet as mx
from mxnet.io import DataDesc
import numpy as np

from util import helper


class DistributedIter(mx.io.DataIter):
    def __init__(self, root, subsets, batch_size = 1, prefetch = True, shuffle = False):
        super(DistributedIter, self).__init__(batch_size)
        self.files = []
        info_files = {}

        for subset in subsets:
            info_files[subset] = helper.read_info_file(os.path.join(root, "subset%d" % subset, "info.txt"))

        check = ["rotate", "flip", "sample_shape", "translate", "type", "resize", "revision"]
        comparison = None
        for subset, info_data in info_files.items():
            if comparison is None:
                comparison = info_data
                continue
            for key, val in comparison.items():
                if key not in check:
                    continue
                if isinstance(val, list):
                    ok = all([x == y for x, y in zip(comparison[key], info_data[key])])
                else:
                    ok = comparison[key] == info_data[key]
                assert ok, "Error: %s is different for subset %d: %s" % (key, subset, str(val))

        self.info = {key: comparison[key] for key in comparison if key in check}

        for subset in subsets:
            files = os.listdir(os.path.join(root, "subset%d" % subset))
            data_files = sorted(filter(lambda x: "data" in x, files))
            label_files = sorted(filter(lambda x: "labels" in x, files))
            for d, l in zip(data_files, label_files):
                self.files.append({
                    "subset": subset,
                    "data": os.path.join(root, "subset%d" % subset, d),
                    "labels": os.path.join(root, "subset%d" % subset, l)})

        if shuffle:
            random.seed(42)
            random.shuffle(self.files)

        self.shuffle = shuffle
        self.prefetch = prefetch
        self.root = root
        self.next_file = 0
        self.__iterator = None
        self.__next_iterator = None

        self.started = True
        self.iterator_ready = threading.Event()
        self.iterator_taken = threading.Event()
        self.iterator_taken.set()

        def prefetch_func(self):
            """Thread entry"""
            while True:
                self.iterator_taken.wait()
                if not self.started:
                    break
                self.__next_iterator = self.load_iterator(self.next_file)
                self.next_file += 1
                self.iterator_taken.clear()
                self.iterator_ready.set()

        self.prefetch_thread = threading.Thread(target = prefetch_func, args = [self])
        self.prefetch_thread.setDaemon(True)
        self.prefetch_thread.start()

    def __del__(self):
        self.started = False
        self.iterator_taken.set()
        self.prefetch_thread.join()

    def sizes(self):
        sizes = {}
        for f in self.files:
            sizes[f["labels"]] = len(np.load(f["labels"]))
        return sizes

    def total_size(self):
        sizes = self.sizes()
        return sum([sizes[x] for x in sizes])

    def __iter__(self):
        return self

    def reset(self):
        self.iterator_ready.wait()
        self.next_file = 0
        self.__iterator = None
        self.__next_iterator = None
        self.iterator_ready.clear()
        self.iterator_taken.set()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [DataDesc("data", tuple([self.total_size()] + self.info["sample_shape"]), helper.DTYPE, "NCDHW")]

    @property
    def provide_label(self):
        return [DataDesc("label", tuple([self.total_size()]), helper.DTYPE, "N")]

    def load_iterator(self, file_nr):
        if file_nr >= len(self.files):
            return None
        data = np.load(self.files[file_nr]["data"], mmap_mode = "r")
        label = np.load(self.files[file_nr]["labels"], mmap_mode = "r")
        return mx.io.NDArrayIter(
            data = data,
            label = label,
            batch_size = self.batch_size,
            shuffle = self.shuffle)

    def take_next_iterator(self):
        self.iterator_ready.wait()
        if self.__next_iterator is None:
            raise StopIteration
        self.__iterator = self.__next_iterator
        self.__next_iterator = None
        self.iterator_ready.clear()
        self.iterator_taken.set()

    def next(self):
        result = None
        while result is None:
            if self.__iterator is None:
                self.take_next_iterator()
            try:
                result = self.__iterator.next()
            except StopIteration:
                self.__iterator = None
        return result


def main(args):
    start = time.time()
    chunk_start = start

    data_iter = DistributedIter(args.root, args.subsets, batch_size = 30)

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

