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
