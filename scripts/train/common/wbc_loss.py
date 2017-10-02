"""
weighted binary-class loss, which is based on cross-entropy loss

"""
import mxnet as mx

def wbc_loss(prob, label, cl_weights):           
    cross_entropy = -(cl_weights[0] * label * mx.symbol.log(prob) + cl_weights[1] * (1 - label) * mx.symbol.log(1 - prob))
    loss = mx.symbol.MakeLoss(cross_entropy, normalization='batch')    
    return loss

