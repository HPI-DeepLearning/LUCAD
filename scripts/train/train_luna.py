import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
import mxnet as mx

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from storage.get_iterator import get_iterator


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser(description="train LUNA16",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # WE SHOULD BUILD AUG funcion in 3D
    data.set_data_aug_level(parser, 3)

    parser.add_argument('--pretrained', type=str,
                    help='the pre-trained model')

    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                    help='save training log to file')

    parser.add_argument('--train-data-root', type=str,
                    help='path to train data subsets')

    parser.add_argument('--val-data-root', type=str,
                    help='path to val data subsets')

    parser.add_argument('--train_subsets', type=str,
                    help='Subsets for training')

    parser.add_argument('--val_subsets', type=str,
                    help='Subsets for validation')
    
    parser.set_defaults(
        # network
        network        = 'mlp',
        num_layers     = 15, # this variable only needed by ResNet

        # data
        num_classes      = 2,
        #num_examples     = 1281167,#imageNet
        image_shape      = '1,36,36,36',#CDHW
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 5,
        lr_step_epochs   = '1,2,3,4',
        lr               = 0.01,
        lr_factor        = 0.5,
        batch_size       = 30,
        optimizer        = 'sgd',
        disp_batches     = 10,
        top_k            = 0,
        train_subsets    = '0,1,2,3,4,5,6,7,8',
        val_subsets      = '9'
    )
    args = parser.parse_args()

    # set up logger
    log_file = args.log_file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]

    #load data
    train_subsets = [int(k) for k in args.train_subsets.split(',')]
    validation_subsets = [int(k) for k in args.val_subsets.split(',')]

    train_iter = get_iterator(args.train_data_root, train_subsets, batch_size = args.batch_size, shuffle = True)
    val_iter = get_iterator(args.val_data_root, validation_subsets, batch_size = args.batch_size)
    args.num_examples = train_iter.total_size()

    # load network
    from importlib import import_module
    net = import_module('3Dsymbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    #load pretrained
    args_params=None
    auxs_params=None

    # train
    if args_params and auxs_params:
        fit.fit(
            args,
            sym,
            train_iter,
            val_iter,
            arg_params=args_params,
            aux_params=auxs_params)
    else:
        fit.fit(
            args,
            sym,
            train_iter,
            val_iter)
