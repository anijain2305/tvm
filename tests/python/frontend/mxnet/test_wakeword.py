import numpy as np
import operator

import tvm
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm import relay
import mxnet as mx

from mxnet import gluon
from mxnet.gluon.model_zoo import vision


def test_fp32():
    import os
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/mxnet-mkl-quantization/mxmet-mkldnn-mnist-convolution-quantization'
    fp32_model_prefix = 'mnist-fp32'
    fp32_model_path = os.path.join(base_path, fp32_model_prefix)

    input_shape = (1, 1, 28, 28)
    input_dict = {'data': input_shape}
    opt_level = 3
    sym, arg_params, aux_params = mx.model.load_checkpoint(fp32_model_path, 0)
    nnvm_sym, nnvm_params = relay.frontend.from_mxnet(sym, shape=input_dict,
                                                      arg_params=arg_params,
                                                      aux_params=aux_params)

def test_qnn():
    import os
    # base_path = '/home/ANT.AMAZON.COM/shoubhik/data/mxnet-mkl-quantization/mxmet-mkldnn-mnist-convolution-quantization'
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/mxnet-mkl-quantization/mxnet-mkldnn-mnist-convolution-without-bias/mnist-conv-quant'
    fp32_model_prefix = 'quantized-mnist-WITH-calibration-and-fusion'
    fp32_model_path = os.path.join(base_path, fp32_model_prefix)

    input_shape = (1, 1, 28, 28)
    input_dict = {'data': input_shape}
    opt_level = 3
    sym, arg_params, aux_params = mx.model.load_checkpoint(fp32_model_path, 0)
    print(sym.debug_str())
    nnvm_sym, nnvm_params = relay.frontend.from_qnn_mxnet(sym,
                                                          dtype='float32',
                                                          shape=input_dict,
                                                          arg_params=arg_params,
                                                          aux_params=aux_params)

def test_conv_with_bias():
    import os
    # base_path = '/home/ANT.AMAZON.COM/shoubhik/data/mxnet-mkl-quantization/mxmet-mkldnn-mnist-convolution-quantization'
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/mxnet-mkl-quantization/mxnet-mkldnn-mnist-convolution-WITH-bias/mnist-conv-quant'
    fp32_model_prefix = 'quantized-mnist-WITH-calibration-and-fusion'
    fp32_model_path = os.path.join(base_path, fp32_model_prefix)

    input_shape = (1, 1, 28, 28)
    input_dict = {'data': input_shape}
    opt_level = 3
    sym, arg_params, aux_params = mx.model.load_checkpoint(fp32_model_path, 0)
    print(sym.debug_str())
    nnvm_sym, nnvm_params = relay.frontend.from_qnn_mxnet(sym,
                                                          dtype='float32',
                                                          shape=input_dict,
                                                          arg_params=arg_params,
                                                          aux_params=aux_params)


def test_quantized_wakeword_model():
    import os
    # base_path = '/home/ANT.AMAZON.COM/shoubhik/data/mxnet-mkl-quantization/mxmet-mkldnn-mnist-convolution-quantization'
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer'
    fp32_model_prefix = 'quantized_wo_fusion_model_wo_ringbuffer'
    fp32_model_path = os.path.join(base_path, fp32_model_prefix)

    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    in_data = np.random.rand(1, 1, 76, 64)
    sym, arg_params, aux_params = mx.model.load_checkpoint(fp32_model_path, 0)
    print(sym.debug_str())
    mod, params = relay.frontend.from_qnn_mxnet(sym,
                                                dtype='float32',
                                                shape=input_dict,
                                                arg_params=arg_params,
                                                aux_params=aux_params)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    with relay.build_config(opt_level=11):
        graph, lib, params = relay.build(mod, "llvm", params=params)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(data=in_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).asnumpy()
        print(res)

if __name__ == '__main__':
    # test_qnn()
    # test_conv_with_bias()
    # test_conv_with_bias()
    test_quantized_wakeword_model()

