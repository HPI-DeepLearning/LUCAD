import argparse
import mxnet as mx
from candidate_iterator import CandidateIter


def main(args):
    batch_size = 30

    train_subsets = (0, 1,)
    validation_subsets = (2,)
    test_subsets = (3,)

    train_iter = CandidateIter(args.root, train_subsets, batch_size = batch_size, shuffle = True)
    val_iter = CandidateIter(args.root, validation_subsets, batch_size = batch_size)

    data = mx.sym.var('data')
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.flatten(data=data)

    # The first fully-connected layer and the corresponding activation function
    fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")

    # The second fully-connected layer and the corresponding activation function
    fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")

    # MNIST has 10 classes
    fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
    # Softmax with cross entropy loss
    mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    import logging
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module on CPU
    mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    mlp_model.fit(train_iter,  # train data
            eval_data=val_iter,  # validation data
            optimizer='sgd',  # use SGD to train
            optimizer_params={'learning_rate':0.1},  # use fixed learning rate
            eval_metric='acc',  # report accuracy during training
            batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
            num_epoch=10)  # train for at most 10 dataset passes

    test_iter = CandidateIter(args.root, test_subsets, batch_size = batch_size)
    acc = mx.metric.Accuracy()
    mlp_model.score(test_iter, acc)
    print(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "train a very simple net")
    parser.add_argument("root", type=str, help="folder containing prepared dataset folders")
    main(parser.parse_args())

