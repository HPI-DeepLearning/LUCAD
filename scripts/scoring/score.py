import argparse
from train.common import find_mxnet
import mxnet as mx
import time
import os
import logging
import sys
import numpy as np
from util import helper

import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from storage.get_iterator import get_iterator

# define classification
cls_labels = ['negative', 'positive']


def score(model_prefix, epoch, val_subsets, metrics, gpus, batch_size, rgb_mean, data_root, output_file, original_data_root, limit, overwrite,
          image_shape='1,36,36,36', data_nthreads=4):

    # create validation iterator
    validation_subsets = [int(k) for k in val_subsets.split(',')]
    val_iter = get_iterator(data_root, validation_subsets, batch_size = batch_size)
    seriesuids = val_iter.get_info()["files"]
    logging.debug("SeriesUIDs: %s" % seriesuids)

    # create output file
    write_output = False
    if output_file != "":
        write_output = True
        parent_directory = os.path.dirname(output_file)
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
        elif os.path.isfile(output_file) and not overwrite:
            logging.error("Not overwriting file without --overwrite.")
            return (0.0,)
        output_handle = open(output_file, "w")
        header = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "class"]
        output_handle.write("%s\n" % ",".join(header))
        # only in this case we need to read the candidates csv
        if original_data_root == "":
            raise ValueError("--original-data-root needs to set to write an output file")
        candidates = helper.load_candidates(original_data_root)
        filtered_data = []
        for s in seriesuids:
            filtered_data += candidates[s]
        logging.debug("Number of Samples: %d" % len(filtered_data))

    # create module
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]
    #prob_sym = sym.get_internals()['softmax_output']
    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(for_training=False,
             data_shapes=val_iter.provide_data,
             label_shapes=val_iter.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]
    logging.info('Info: model scoring started...')
    total_bat = 0
    num = 0
    tic = time.time()

    confusion = np.zeros((2, 2))

    for batch in val_iter:
        mod.forward(batch, is_train=False)
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        for i, p in enumerate(prob):
            a = np.argmax(p)
            current_label = int(round(batch.label[0].asnumpy()[i]))
            confusion[current_label, a] += 1
            if write_output:
                if (num + i) >= len(filtered_data):
                    logging.debug("Skipping %d, padded batch" % (num + i))
                    logging.debug("Batch padding: %s, Index: %d" % (str(batch.pad), i))
                    break
                assert current_label == int(filtered_data[num + i]["class"]), "original and processed labels not equal"
                filtered_data[num + i]["probability"] = p[1]
                filtered_data[num + i]["class"] = a
                output_handle.write("%s\n" % ",".join([str(filtered_data[num + i][col]) for col in header]))
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if 0 < limit <= num:
            total_bat = time.time() - tic
            logging.info('%f second per image, total time: %f', total_bat/num, total_bat)
            break
    if write_output:
        output_handle.close()
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
    parser.add_argument('--model-prefix', type=str, required=True,
                        help = 'the model prefix.')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--val-subsets', type=str, required=True)
    parser.add_argument('--image-shape', type=str, default='1,36,36,36')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-file', type=str, default="")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--data-nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    parser.add_argument('--epoch', type=int, default=0,
                        help='epoch of the model')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--original-data-root', type=str, default = "")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc'), mx.metric.create('f1')]#,
               #mx.metric.create('top_k_accuracy', top_k = 5)]

    (speed,) = score(metrics = metrics, **vars(args))
    logging.info('Finished with %f images per second', speed)
    for m in metrics:
        logging.info(m.get())
