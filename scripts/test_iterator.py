import os
import mxnet as mx
import numpy as np


def get_test_iterator(batch_size = 30):
    data = np.load(os.path.join("test_data", "data.npy"))
    label = np.load(os.path.join("test_data", "labels.npy"))
    return mx.io.NDArrayIter(data = data, label = label, batch_size = batch_size)

def main():
    data_iter = get_test_iterator(30)
    for batch in data_iter:
        print([batch.data, batch.label, batch.pad])

if __name__ == "__main__":
    main()

