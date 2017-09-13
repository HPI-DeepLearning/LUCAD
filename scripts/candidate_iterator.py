import mxnet as mx
import os
import subprocess
import numpy as np
import tarfile

data = np.random.rand(100,3,2,2)
label = np.random.randint(0, 2, (100,))
data_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=30)
for batch in data_iter:
    print([batch.data, batch.label, batch.pad])


class CandidateIter(mx.io.DataIter):
    def __init__(self, root, subsets, batch_size = 1, data_name = 'data', label_name = 'softmax_label'):
        self.data_files = []
        self.label_files = []
        for subset in subsets:
            with open(os.path.join(root, "subset%d.txt" % subset)) as handle:
                for line in handle:
                    split = line.split(" ")

                    assert "data" in split[0]
                    self.data_files.append(split[0])

                    assert "label" in split[1]
                    self.label_files.append(split[1])

        self.root = root
        self.batch_size = batch_size
        self.data_name = data_name
        self.label_name = label_name
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.current_file = 0
        self.current_file_data = None
        self.current_file_labels = None

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return False

    @property
    def provide_label(self):
        return False

    def next(self):
        if self.current_file_data == None:
            if self.current_file == len(self.data_files):
                raise StopIteration

            # test memory mapping?
            self.current_file_data = np.load(os.path.join(self.root, self.data_files[self.current_file]))
            self.current_file_labels = np.load(os.path.join(self.root, self.label_files[self.current_file]))

        if self.current_file < len(self.data_files):
            self.current_file += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration

def main(args):
    c = CandidateIter(args.root, args.subsets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test the dataset iterator")
    parser.add_argument("root", type=str, help="folder containing prepared dataset folders")
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = range(0, 10))
    main(parser.parse_args())

