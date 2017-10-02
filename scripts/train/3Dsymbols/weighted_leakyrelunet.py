"""
A simple solution for Lung Nodule classification
Reference:
The 2nd place solution to theNational Data Science Bowl 2017 hosted by Kaggle.com
https://github.com/juliandewit/kaggle_ndsb2017
"""
import mxnet as mx
from common.wbc_loss import wbc_loss
from common.math_ops import *

eps = 1e-10 + 1e-5
bn_mom = 0.9
fix_gamma = False

def get_symbol(class_weights,num_classes, **kwargs):
    input_data = mx.symbol.Variable(name="data")

    #pool0 = mx.symbol.Pooling(
    #    data=input_data, pool_type="avg", kernel=(2, 1, 1), stride=(2, 1, 1),)
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(3, 3, 3), stride=(1, 1, 1), pad=(1,1,1), num_filter=64)
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu1 = mx.symbol.LeakyReLU(data=bn1, act_type="leaky")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2))
    
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3, 3), stride=(1, 1, 1), num_filter=128)
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu2 = mx.symbol.LeakyReLU(data=bn2, act_type="leaky")
    pool2 = mx.symbol.Pooling(
        data=relu2, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2))
    
    # stage 3
    conv3a = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3, 3), stride=(1, 1, 1), pad=(1,1,1), num_filter=256)
    bn3a = mx.sym.BatchNorm(data=conv3a, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu3a = mx.symbol.LeakyReLU(data=bn3a, act_type="leaky")
    conv3b = mx.symbol.Convolution(
        data=relu3a, kernel=(3, 3, 3), stride=(1, 1, 1), pad=(1,1,1), num_filter=256)
    bn3b = mx.sym.BatchNorm(data=conv3b, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu3b = mx.symbol.LeakyReLU(data=bn3b, act_type="leaky")

    pool3 = mx.symbol.Pooling(
        data=relu3b, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2))

    dropout1 = mx.symbol.Dropout(pool3, p = 0.5)

    # stage 4
    conv4a = mx.symbol.Convolution(
        data=dropout1, kernel=(3, 3, 3), stride=(1, 1, 1), pad=(1,1,1), num_filter=512)
    bn4a = mx.sym.BatchNorm(data=conv4a, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu4a = mx.symbol.LeakyReLU(data=bn4a, act_type="leaky")
    conv4b = mx.symbol.Convolution(
        data=relu4a, kernel=(3, 3, 3), stride=(1, 1, 1), pad=(1,1,1), num_filter=512)
    bn4b = mx.sym.BatchNorm(data=conv4b, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu4b = mx.symbol.LeakyReLU(data=bn4b, act_type="leaky")

    pool4 = mx.symbol.Pooling(
        data=relu4b, pool_type="max", kernel=(2, 2, 2), stride=(2, 2, 2))

    dropout2 = mx.symbol.Dropout(pool4, p = 0.5)

    # stage 5
    conv5 = mx.symbol.Convolution(
        data=dropout2, kernel=(2, 2, 2), stride=(1, 1, 1), num_filter=64)
    bn5 = mx.sym.BatchNorm(data=conv5, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu5 = mx.symbol.LeakyReLU(data=bn5, act_type="leaky")    

    dropout3 = mx.symbol.Dropout(relu5, p = 0.5)

    # stage 6
    flatten = mx.symbol.Flatten(data=dropout3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name="fc_pred")

    # stage 7 create weighted loss for binary classification
    label = mx.symbol.Variable('softmax_label')
    label = mx.symbol.reshape(data=label, shape=(0,1))
    label = mx.symbol.broadcast_axis(data=label, axis=1, size=2)    
    #label = mx.sym.Custom(data=label, op_type='debug')

    prob = mx.symbol.softmax(data=fc1, name="softmax")
    #prob = mx.sym.Custom(data=prob, op_type='debug')
    loss = wbc_loss(prob=prob, label=label, cl_weights = get_class_weights(class_weights))
    
    #pred_loss = mx.symbol.Group([mx.symbol.BlockGrad(out), loss])
    #arg_shape, output_shape, aux_shape = out.infer_shape(data=(5, 1, 36,36,36))
    #print "output shape:" + str(output_shape)
    return loss    

def get_class_weights(class_weights):
    cl_weights = list()
    for idx, w in enumerate(class_weights.split(',')):
        cl_weights.append(float(w))
    print "Class weights: " + str(cl_weights)
    return cl_weights