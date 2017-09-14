import os
import mxnet as mx
import numpy as np


def get_test_iterators(batch_size = 30):
    data = np.load(os.path.join("test_data", "data.npy"))
    label = np.load(os.path.join("test_data", "labels.npy"))

    # add additional channel dimension
    data = np.expand_dims(data, axis = 1)

    train_data = data[:-180]
    train_label = label[:-180]
    # print float(sum(train_label)) / len(train_label)

    train_iter = mx.io.NDArrayIter(data = train_data, label = train_label, batch_size = batch_size, shuffle = True)

    validation_data = data[-180:-90]
    validation_label = label[-180:-90]
    # print float(sum(validation_label)) / len(validation_label)

    validation_iter = mx.io.NDArrayIter(data = validation_data, label = validation_label, batch_size = batch_size)

    test_data = data[-90:]
    test_label = label[-90:]
    # print float(sum(test_label)) / len(test_label)

    test_iter = mx.io.NDArrayIter(data = test_data, label = test_label, batch_size = batch_size)

    return train_iter, validation_iter, test_iter


def main():
    train_iter, validation_iter, test_iter = get_test_iterators(30)
    for batch in train_iter:
        print "Batch shape: %s" % ",".join([str(x) for x in batch.data[0].shape])


if __name__ == "__main__":
    main()

