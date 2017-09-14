import argparse, sys, os
import numpy as np
sys.path.append("scripts")
from candidate_iterator import CandidateIter
import helper


def main(args):
    iterator = CandidateIter(args.root, (args.subset,), batch_size = args.batchsize, shuffle = True)
    data_shape = iterator.provide_data[0].shape
    data_shape = (args.n * args.batchsize, data_shape[1], data_shape[2], data_shape[3])

    data = np.zeros(data_shape, dtype = helper.DTYPE)
    label = np.zeros(args.n * args.batchsize, dtype = helper.DTYPE)

    lb = helper.SimpleLoadingBar("Extracting", args.n)
    for i in range(0, args.n):
        batch = iterator.next()
        start = i * args.batchsize
        end = (i + 1) * args.batchsize
        data[start:end] = batch.data[0].asnumpy()
        label[start:end] = batch.label[0].asnumpy()
        lb.update_progress(i)

    np.save(os.path.join(args.output, "data.npy"), data)
    np.save(os.path.join(args.output, "labels.npy"), label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "extract small portion of a prepared 3D array")
    parser.add_argument("root", type=str, help="folder containing prepared dataset folders")
    parser.add_argument("output", type=str, nargs="?", help="output folder", default = "test_data")
    parser.add_argument("--n", type=int, help="number of batches to extract", default = 10)
    parser.add_argument("--subset", type=int, help="the subset which should be used", default = 0)
    parser.add_argument("--batchsize", type=int, help="batch size", default = 30)
    main(parser.parse_args())
