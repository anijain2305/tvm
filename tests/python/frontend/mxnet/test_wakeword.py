import numpy as np
import operator

import tvm
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime
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

def run_intermediate_quantized_layers():
    import os
    import numpy as np
    import mxnet as mx
    import tvm
    from collections import namedtuple
    from tvm.contrib import graph_runtime
    from tvm import relay

    Batch = namedtuple('Batch', ['data'])
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs'
    ww_quantized_model = 'quantized_wo_fusion_model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    input_data = np.load(base_path +"/input_data.npy")
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)
    # print(sym.debug_str())
    all_layers = sym.get_internals()
    print(all_layers.list_outputs())
    layer_to_output = 'quantized_sg_mkldnn_conv_1_output'
    sym3 = all_layers[layer_to_output]
    mod3 = mx.mod.Module(symbol=sym3, label_names=None, context=mx.cpu())
    mod3.bind(for_training=False, data_shapes=[('data', input_shape)])
    mod3.set_params(arg_params, aux_params)
    mod3.forward(Batch([mx.nd.array(input_data)]))
    out = mod3.get_outputs()[0].asnumpy()
    # print(out)
    mod3.save_checkpoint('{}'.format(layer_to_output), 0)
    np.save('{}_out.npy'.format(layer_to_output), out)
    mod, params = relay.frontend.from_qnn_mxnet(sym3,
                                                dtype='float32',
                                                shape=input_dict,
                                                arg_params=arg_params,
                                                aux_params=aux_params)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print(mod)
    with relay.build_config(opt_level=11):
        graph, lib, params = relay.build(mod, "llvm", params=params)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(data=input_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).asnumpy()
        print(res)



if __name__ == '__main__':
    # test_qnn()
    # test_conv_with_bias()
    # test_conv_with_bias()
    # test_quantized_wakeword_model()
    run_intermediate_quantized_layers()
