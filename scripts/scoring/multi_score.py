import argparse
from train.common import find_mxnet
import mxnet as mx
import time
import os
import logging
import sys
import numpy as np
from util import helper
from util.run_loader import MultiRunLoader

import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from storage.get_iterator import get_iterator
from util.config_init import LUCADConfig

# define classification
cls_labels = ['negative', 'positive']


def get_dir():
    return os.path.join(c.str("parent_dir"), c.str("dir").format(timestamp=helper.now()))


class Scorer(object):
    def __init__(self):
        self.loaders = []
        self.weights = []
        self.epochs = []
        self.validation_subsets = None
        for model_index in range(0, 10):
            model = "model%d" % model_index
            if not c.config.has_option("models", model):
                continue
            loader = MultiRunLoader(c.str(model))
            if len(loader.runs) > 0:
                self.loaders.append(loader)
                self.weights.append(c.float("weight%d" % model_index))
                self.epochs.append(c.str("epoch%d" % model_index))
                if self.validation_subsets is None:
                    self.validation_subsets = sorted(loader.runs.keys())
                else:
                    self.validation_subsets = filter(lambda x: x in loader.runs.keys(), self.validation_subsets)
        if c.str("val_subsets") != "AUTO":
            self.validation_subsets = [int(k) for k in c.str("val_subsets").split(',')]
        self.write_output = False

    def score_all(self):
        for val_set in self.validation_subsets:
            (speed,) = self.score(val_set)
            logging.info('Finished with %f images per second', speed)

    def setup_output(self, subset):
        if c.str("file") != "":
            output_file = os.path.join(get_dir(), c.str("file").format(val_subset=subset))
            self.write_output = True
            if not os.path.exists(get_dir()):
                os.makedirs(get_dir())
            elif os.path.isfile(output_file) and not c.bool("overwrite"):
                logging.error("Not overwriting file without --overwrite.")
                return (0.0,)
            self.output_handle = open(output_file, "w")
            self.header = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "class"]
            self.output_handle.write("%s\n" % ",".join(self.header))
            # only in this case we need to read the candidates csv
            if not os.path.isdir(c.str("original_data_root")):
                raise ValueError("--original-data-root is not a directory")
            candidates = helper.load_candidates(c.str("original_data_root"))
            self.filtered_data = []
            for s in self.seriesuids:
                self.filtered_data += candidates[s]
            logging.debug("Number of Samples: %d" % len(self.filtered_data))

    def load_models(self, subset):
        # create modules
        self.modules = []
        for i in range(0, len(self.loaders)):
            sym, arg_params, aux_params, epoch = self.loaders[i].load(subset, self.epochs[i])
            logging.info("Loaded epoch %d of model %s" % (epoch, self.loaders[i].runs[subset].prefix))
            if c.str("gpus") == '':
                context = mx.cpu()
            else:
                context = [mx.gpu(int(i)) for i in c.str("gpus").split(',')]
            mod = mx.mod.Module(symbol=sym, context=context)
            mod.bind(for_training=False,
                     data_shapes=self.val_iter.provide_data,
                     label_shapes=self.val_iter.provide_label)
            mod.set_params(arg_params, aux_params)
            self.modules.append(mod)

    def score(self, subset):
        # create validation iterator
        self.val_iter = get_iterator(c.str("data_root"), [subset], batch_size=c.int("batch_size"))
        self.seriesuids = self.val_iter.get_info()["files"]
        logging.debug("SeriesUIDs: %s" % self.seriesuids)

        self.setup_output(subset)
        self.load_models(subset)

        logging.info('Info: model scoring started...')
        num = 0
        tic = time.time()

        confusion = np.zeros((2, 2))

        for batch in self.val_iter:
            self.probs = []
            for i, mod in enumerate(self.modules):
                mod.forward(batch, is_train=False)
                prob = mod.get_outputs()[0].asnumpy() * self.weights[i]
                self.probs.append(np.expand_dims(prob, 0))
            prob = np.mean(np.vstack(self.probs), axis=0) / sum(self.weights)
            for i, p in enumerate(prob):
                a = np.argmax(p)
                current_label = int(round(batch.label[0].asnumpy()[i]))
                confusion[current_label, a] += 1
                if self.write_output:
                    if (num + i) >= len(self.filtered_data):
                        logging.debug("Skipping %d, padded batch" % (num + i))
                        logging.debug("Batch padding: %s, Index: %d" % (str(batch.pad), i))
                        break
                    assert current_label == int(self.filtered_data[num + i]["class"]), "original and processed labels not equal"
                    self.filtered_data[num + i]["probability"] = p[1]
                    self.filtered_data[num + i]["class"] = a
                    self.output_handle.write("%s\n" % ",".join([str(self.filtered_data[num + i][col]) for col in self.header]))
            num += c.int("batch_size")
            if 0 < c.int("limit") <= num:
                total_bat = time.time() - tic
                logging.info('%f second per image, total time: %f', total_bat/num, total_bat)
                break
        if self.write_output:
            self.output_handle.close()
        logging.info("True  Negatives (Label 0, Predicted 0): %d" % confusion[0][0])
        logging.info("False Positives (Label 0, Predicted 1): %d" % confusion[0][1])
        logging.info("True  Positives (Label 1, Predicted 1): %d" % confusion[1][1])
        logging.info("False Negatives (Label 1, Predicted 0): %d" % confusion[1][0])

        precision = confusion[1][1]/(confusion[0][1]+confusion[1][1])
        recall = confusion[1][1]/(confusion[1][0]+confusion[1][1])
        f1 = 2 * precision * recall / (precision + recall)
        logging.info("Precision: %f" % precision)
        logging.info("Recall: %f" % recall)
        logging.info("F1-total: %f" % f1)
        return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--val-subsets', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action="store_true", default=None)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    c = LUCADConfig(args = args)

    path = get_dir()
    if not os.path.exists(path):
        os.makedirs(path)

    if c.str("log") != "":
        log_file = os.path.join(path, c.str("log"))
        handler = logging.FileHandler(log_file, mode='w')
        handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logger.addHandler(handler)

    scorer = Scorer()
    scorer.score_all()
    c.write(os.path.join(path, "ms-config.ini"))
