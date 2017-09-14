import os
import mxnet as mx
import numpy as np


class TestIter(mx.io.DataIter):
    def __init__(self, *args, **kwargs):
        self.iterator = mx.io.NDArrayIter(*args, **kwargs)

    def __iter__(self):
        return self

    def reset(self):
        self.iterator.reset()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        data = self.iterator.provide_data
        data[0].layout = "NCDHW"
        return data

    @property
    def provide_label(self):
        return self.iterator.provide_label

    def next(self):
        return self.iterator.next()


def get_test_iterators(batch_size = 30):
    data = np.load(os.path.join("test_data", "data.npy"))
    label = np.load(os.path.join("test_data", "labels.npy"))

    # add additional channel dimension
    data = np.expand_dims(data, axis = 1)

    train_data = data[:-180]
    train_label = label[:-180]
    # print float(sum(train_label)) / len(train_label)

    train_iter = TestIter(data = train_data, label = train_label, batch_size = batch_size, shuffle = True)

    validation_data = data[-180:-90]
    validation_label = label[-180:-90]
    # print float(sum(validation_label)) / len(validation_label)

    validation_iter = TestIter(data = validation_data, label = validation_label, batch_size = batch_size)

    test_data = data[-90:]
    test_label = label[-90:]
    # print float(sum(test_label)) / len(test_label)

    test_iter = TestIter(data = test_data, label = test_label, batch_size = batch_size)

    return train_iter, validation_iter, test_iter


def main():
    train_iter, validation_iter, test_iter = get_test_iterators(30)
    print train_iter.provide_data[0].layout
    print train_iter.provide_data[0].shape
    for batch in train_iter:
        print "Batch shape: %s" % ",".join([str(x) for x in batch.data[0].shape])


if __name__ == "__main__":
    main()

