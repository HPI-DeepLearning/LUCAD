import os
import logging

from util import helper
from candidate_iterator import CandidateIter
from distributed_iterator import DistributedIter
from mxnet.io import PrefetchingIter


def get_iterator(root, subsets, batch_size, chunk_size = 100, shuffle = False, prefetch = False, data_name = 'data', label_name = 'softmax_label'):
    info = helper.read_info_file(os.path.join(root, "subset%d" % subsets[0], "info.txt"))
    if "type" not in info or info["type"] == "CandidateStorage":
        return CandidateIter(root, subsets, batch_size = batch_size, shuffle = shuffle, chunk_size = chunk_size, data_name = data_name, label_name = label_name)
    if info["type"] == "DistributedStorage":
        return DistributedIter(root, subsets, batch_size = batch_size, prefetch = prefetch, shuffle = shuffle, data_name = data_name, label_name = label_name)
    return None


def main(args):
    start = time.time()
    chunk_start = start

    batch_size = 30

    data_iter = get_iterator(args.root, args.subsets, batch_size = batch_size, shuffle = True, chunk_size = 100)

    print "Sizes: %s" % data_iter.sizes()
    print "Total number of samples: %d" % data_iter.total_size()
    print "Data layout: %s" % data_iter.provide_data[0].layout
    print "Data shape: %s" % str(data_iter.provide_data[0].shape)
    print "Number of batches: ~%d" % (data_iter.total_size() / batch_size)

    measure_chunks = 1000

    i = 0
    # lb = helper.SimpleLoadingBar("Loading samples", data_iter.total_size())
    for batch in data_iter:
        if i == 0:
            print "Batch shape: %s" % ",".join([str(x) for x in batch.data[0].shape])
        if i % measure_chunks == (measure_chunks - 1):
            prev_chunk = chunk_start
            chunk_start = time.time()
            chunk = chunk_start - prev_chunk
            avg = (chunk_start - start) / ((i+1) / float(measure_chunks))
            print "Batch %d (%.2f, total avg: %.2f sec / %d batch)!" % (i + 1, chunk, avg, measure_chunks)
        assert len(batch.data[0]) == batch_size
        assert len(batch.label[0]) == batch_size
        # lb.advance_progress(batch_size)
        i += 1
    print "Finished!"


if __name__ == "__main__":
    import sys
    import argparse
    import time

    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)

    parser = argparse.ArgumentParser(description = "test the dataset iterator")
    parser.add_argument("root", type=str, help="folder containing prepared dataset folders")
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = range(0, 10))
    main(parser.parse_args())
