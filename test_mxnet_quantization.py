import numpy as np
import operator

import tvm
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as debug_runtime
from tvm import relay
import mxnet as mx
from tvm.contrib import ndk
import time

from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import statistics
# Command for EchoDot Gen 2 - dlr/bin/model_executor models/dlr_tf_mobilenet_v1_100 cat224-3.npy cpu input

def resnet50_quantized():
    import os
    import numpy as np
    import mxnet as mx
    import tvm
    from collections import namedtuple
    from tvm.contrib import graph_runtime
    from tvm import relay
    from os import listdir
    from os.path import isfile, join
    import glob

    Batch = namedtuple('Batch', ['data'])
    base_path = '/home/ubuntu/mxnet/incubator-mxnet/example/quantization/model'
    ww_quantized_model = 'resnet50_v1-quantized-5batches-naive'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 3, 224, 224)
    input_dict = {'data': input_shape}
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)

    mod, params = relay.frontend.from_qnn_mxnet(sym,
                                            dtype='float32',
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)
    print(mod)

    target = 'llvm -mcpu=cascadelake'
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
        base = '/home/ubuntu/mxnet_compiled_models/'
        ww_quantized_model = 'resnet50_v1-quantized-5batches-naive'

        path_lib = base + ww_quantized_model + '_deploy_lib.tar'
        path_graph =  base + ww_quantized_model + '_deploy_graph.json'
        path_params = base + ww_quantized_model + '_deploy_params.params'

        lib.export_library(path_lib)
        with open(path_graph, 'w') as fo:
            fo.write(graph)
        with open(path_params, 'wb') as fo:
            fo.write(relay.save_param_dict(params))

        rt_mod = debug_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(**params)
        rt_mod.run()


def resnet50_profile():
    base = '/home/ubuntu/mxnet_compiled_models/'
    ww_quantized_model = 'resnet50_v1-quantized-5batches-naive'
    path_lib = base + ww_quantized_model + '_deploy_lib.tar'
    path_graph =  base + ww_quantized_model + '_deploy_graph.json'
    path_params = base + ww_quantized_model + '_deploy_params.params'

    graph = open(path_graph).read()
    lib = tvm.module.load(path_lib)
    params = bytearray(open(path_params, 'rb').read())

    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.load_params(params)
    # rt_mod.set_input('data', batch.data[0].asnumpy())
    for i in range(0, 10):
        rt_mod.run()

    num_iters = 10000
    total = 0
    runtimes = list()
    for i in range(0, num_iters):
        t1 = time.time()
        rt_mod.run()
        t2 = time.time()
        total = t2 - t1
        runtimes.append(total)

    print(statistics.mean(runtimes))
    print(statistics.stdev(runtimes))

    

if __name__ == '__main__':
    resnet50_quantized()
    resnet50_profile()
