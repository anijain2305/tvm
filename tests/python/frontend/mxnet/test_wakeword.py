import numpy as np
import operator

import tvm
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime
from tvm import relay
import mxnet as mx
from tvm.contrib import ndk

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
    # base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs'
    # ww_quantized_model = 'quantized_wo_fusion_model_wo_ringbuffer'
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/model-with-fused-activation'
    ww_quantized_model = 'quantized_w_fusion_model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    input_data = np.load(base_path +"/input_data.npy")
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)
    # print(sym.debug_str())
    all_layers = sym.get_internals()
    print(all_layers.list_outputs())
    # layer_to_output = 'quantized_sg_mkldnn_conv_act_1_output'
    layer_to_output = 'quantized_sg_mkldnn_fully_connected_3_output'
    # layer_to_output = 'quantized_sg_mkldnn_fully_connected_relu_2_output'
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

    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, "llvm", params=params)
        # print(graph)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(data=input_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).asnumpy()
        print(res)
        print(np.count_nonzero(res))

def ww_fp32_model_local():
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
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs'
    ww_quantized_model = 'model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)

    mod, params = relay.frontend.from_mxnet(sym,
                                            dtype='float32',
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print(mod)
    target = 'llvm'
    validation_data_folder = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/validataion-data/archive/'
    validation_data_files = [f for f in glob.glob(validation_data_folder + "*.npy")]
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(**params)
        for validation_data in validation_data_files:
            preds = []
            pred_file_name = validation_data
            label = 'True' if 'True' in validation_data else 'False'
            validation_data = np.load(validation_data)
            feats = np.reshape(validation_data, (1, 1, -1, 64))
            for k in range(4, np.shape(feats)[2] - 5, 6):
                if k >= 70:
                    input_data = feats[:, :, k-70:k + 6, :]
                    rt_mod.set_input(data=input_data)
                    rt_mod.run()
                    res = rt_mod.get_output(0).asnumpy()
                    preds.append(res)
                    # print('{}: {}'.format(label, res))
            preds = np.concatenate(preds)
            print('validating for {}'.format(pred_file_name))
            if label == 'True':
                assert max(preds[:, 1]) > 0.8, max(preds[:, 1])
            else:
                assert max(preds[:, 1]) < 0.3, max(preds[:, 1])

def ww_fp32_mxnet_model_local_accuracy_validation():
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
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs'
    ww_quantized_model = 'model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)

    mod3 = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
    mod3.bind(for_training=False, data_shapes=[('data', input_shape)])
    mod3.set_params(arg_params, aux_params)
    validation_data_folder = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/validataion-data/archive/'
    validation_data_files = [f for f in glob.glob(validation_data_folder + "*.npy")]
    total_predictions = 0
    correct_predictions = 0
    incorrect_predictions = 0
    for validation_data in validation_data_files:
        preds = []
        pred_file_name = validation_data
        label = 'True' if 'True' in validation_data else 'False'
        validation_data = np.load(validation_data)
        feats = np.reshape(validation_data, (1, 1, -1, 64))
        for k in range(4, np.shape(feats)[2] - 5, 6):
            if k >= 70:
                input_data = feats[:, :, k-70:k + 6, :]
                mod3.forward(Batch([mx.nd.array(input_data)]))
                out = mod3.get_outputs()[0].asnumpy()
                preds.append(out)
        mx_preds = np.concatenate(preds)
        print('validating for {}'.format(pred_file_name))
        if label == 'True':
            # if(max(mx_preds[:, 1]) > 0.8):
            #     correct_predictions += 1
            # else:
            #     incorrect_predictions += 1
            assert max(mx_preds[:, 1]) > 0.8, max(preds[:, 1])
        else:
            # if max(mx_preds[:, 1]) < 0.3:
            #     correct_predictions += 1
            # else:
            #     incorrect_predictions += 1
            assert max(mx_preds[:, 1]) < 0.3, max(preds[:, 1])
    print('total predictions:{}'.format(total_predictions))
    print('correct predictions:{}'.format(correct_predictions))
    print('incorrect predictions:{}'.format(incorrect_predictions))


def ww_int8_mxnet_model_local_accuracy_validation():
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
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs'
    ww_quantized_model = 'quantized_wo_fusion_model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)

    mod3 = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
    mod3.bind(for_training=False, data_shapes=[('data', input_shape)])
    mod3.set_params(arg_params, aux_params)
    validation_data_folder = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/validataion-data/archive/'
    validation_data_files = [f for f in glob.glob(validation_data_folder + "*.npy")]
    total_predictions = 0
    correct_predictions = 0
    incorrect_predictions = 0
    for validation_data in validation_data_files:
        preds = []
        pred_file_name = validation_data
        label = 'True' if 'True' in validation_data else 'False'
        validation_data = np.load(validation_data)
        feats = np.reshape(validation_data, (1, 1, -1, 64))
        for k in range(4, np.shape(feats)[2] - 5, 6):
            if k >= 70:
                input_data = feats[:, :, k-70:k + 6, :]
                mod3.forward(Batch([mx.nd.array(input_data)]))
                out = mod3.get_outputs()[0].asnumpy()
                preds.append(out)
        mx_preds = np.concatenate(preds)
        print('validating for {}'.format(pred_file_name))
        if label == 'True':
            # if(max(mx_preds[:, 1]) > 0.8):
            #     correct_predictions += 1
            # else:
            #     incorrect_predictions += 1
            assert max(mx_preds[:, 1]) > 0.8, max(preds[:, 1])
        else:
            # if max(mx_preds[:, 1]) < 0.3:
            #     correct_predictions += 1
            # else:
            #     incorrect_predictions += 1
            assert max(mx_preds[:, 1]) < 0.3, max(preds[:, 1])
    print('total predictions:{}'.format(total_predictions))
    print('correct predictions:{}'.format(correct_predictions))
    print('incorrect predictions:{}'.format(incorrect_predictions))


def ww_int8_model_local_accuracy_validation():
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
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs/quantized-tuned-model/'
    ww_quantized_model = 'quantized_wo_fusion_model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)

    mod, params = relay.frontend.from_qnn_mxnet(sym,
                                            dtype='float32',
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print(mod)
    target = 'llvm'
    validation_data_folder = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/validataion-data/archive/'
    validation_data_files = [f for f in glob.glob(validation_data_folder + "*.npy")]
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(**params)
        total_predictions = 0
        correct_predictions = 0
        incorrect_predictions = 0
        for validation_data in validation_data_files:
            preds = []
            pred_file_name = validation_data
            label = 'True' if 'True' in validation_data else 'False'
            validation_data = np.load(validation_data)
            feats = np.reshape(validation_data, (1, 1, -1, 64))
            for k in range(4, np.shape(feats)[2] - 5, 6):
                if k >= 70:
                    input_data = feats[:, :, k-70:k + 6, :]
                    rt_mod.set_input(data=input_data)
                    rt_mod.run()
                    res = rt_mod.get_output(0).asnumpy()
                    preds.append(res)
                    total_predictions += 1
                    # print('{}: {}'.format(label, res))
            preds = np.concatenate(preds)
            print('validating for {}'.format(pred_file_name))
            if label == 'True':
                if(max(preds[:, 1]) > 0.8):
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                # assert max(preds[:, 1]) > 0.8, max(preds[:, 1])
            else:
                if max(preds[:, 1]) < 0.3:
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                # assert max(preds[:, 1]) < 0.3, max(preds[:, 1])
        print('total predictions:{}'.format(total_predictions))
        print('correct predictions:{}'.format(correct_predictions))
        print('incorrect predictions:{}'.format(incorrect_predictions))



def cross_compile_for_fp32_model_echo_dot2():
    import os
    import numpy as np
    import mxnet as mx
    import tvm
    from collections import namedtuple
    from tvm.contrib import graph_runtime
    from tvm import relay

    Batch = namedtuple('Batch', ['data'])
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/intermediat-layer-outputs'
    ww_quantized_model = 'model_wo_ringbuffer'
    ww_quantized_model = os.path.join(base_path, ww_quantized_model)
    input_shape = (1, 1, 76, 64)
    input_dict = {'data': input_shape}
    input_data = np.load(base_path +"/input_data.npy")
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)
    # print(sym.debug_str())

    mod, params = relay.frontend.from_mxnet(sym,
                                            dtype='float32',
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print(mod)
    # For armv7a (32bits)
    # with TVM_NDK_CC=/Users/rankyunh/arm-linux-toolchain/bin/arm-linux-androideabi-g++
    target = 'llvm -target=armv7a-none-linux-android -mfloat-abi=soft -mattr=+neon -mcpu=cortex-a53'
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
        output_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/tvm-android-model/tvm-crosscompiled-fp32-sputnik-model/'
        path_so = output_path + "android_lib_fp32.so"
        path_graph = output_path + "android_graph_fp32.json"
        path_param = output_path + "android_param_fp32.params"
        # lib.export_library(path_so, ndk.create_shared)
        cc = '/home/ANT.AMAZON.COM/shoubhik/android-toolchain/bin/arm-linux-androideabi-clang++'
        lib.export_library(path_so, cc=cc, options=['-static-libstdc++'])
        with open(path_graph, "w") as fo:
            fo.write(graph)
        with open(path_param, "wb") as fo:
            fo.write(relay.save_param_dict(params))
        print("check files in %s" % output_path)


def cross_compile_for_echo_dot2():
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
    # input_shape = (1,3 , 224, 224)
    input_dict = {'data': input_shape}
    input_data = np.load(base_path +"/input_data.npy")
    sym, arg_params, aux_params = mx.model.load_checkpoint(ww_quantized_model, 0)
    # print(sym.debug_str())

    mod, params = relay.frontend.from_qnn_mxnet(sym,
                                                dtype='float32',
                                                shape=input_dict,
                                                arg_params=arg_params,
                                                aux_params=aux_params)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print(mod)
    # For armv7a (32bits)
    # with TVM_NDK_CC=/Users/rankyunh/arm-linux-toolchain/bin/arm-linux-androideabi-g++
    target = 'llvm -target=armv7a-none-linux-android -mfloat-abi=soft -mattr=+neon -mcpu=cortex-a53'
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
        output_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/tvm-android-model/'
        path_so = output_path + "android_lib.so"
        path_graph = output_path + "android_graph.json"
        path_param = output_path + "android_param.params"
        # lib.export_library(path_so, ndk.create_shared)
        cc = '/home/ANT.AMAZON.COM/shoubhik/android-toolchain/bin/arm-linux-androideabi-clang++'
        lib.export_library(path_so, cc=cc, options=['-static-libstdc++'])
        with open(path_graph, "w") as fo:
            fo.write(graph)
        with open(path_param, "wb") as fo:
            fo.write(relay.save_param_dict(params))
        print("check files in %s" % output_path)


def run_intermediate_quantized_layers_uint8():
    import os
    import numpy as np
    import mxnet as mx
    import tvm
    from collections import namedtuple
    from tvm.contrib import graph_runtime
    from tvm import relay

    Batch = namedtuple('Batch', ['data'])
    base_path = '/home/ANT.AMAZON.COM/shoubhik/data/wakeword/sputnik-mxnet-model-wo-ring-buffer/uint8-quantization'
    ww_quantized_model = 'quantized_model_wo_ringbuffer_w_fusion'
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
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, "llvm", params=params)
        # print(graph)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(data=input_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).asnumpy()
        print(res)


from mxnet.contrib.quantization import *
def quantize_ww_model_with_uint8():
    import mxnet as mx
    import logging
    from mxnet.contrib.quantization import _calibrate_quantized_sym, _quantize_params, _quantize_symbol, \
        _LayerOutputCollector, _LayerOutputMinMaxCollector

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    def quantize_graph(sym, arg_params, aux_params,
                       excluded_sym_names=None, calib_mode='entropy',
                       calib_layer=None, quantized_dtype='int8', logger=logging):

        if excluded_sym_names is None:
            excluded_sym_names = []
        if not isinstance(excluded_sym_names, list):
            raise ValueError('excluded_sym_names must be a list of strings representing'
                             ' the names of the symbols that will not be quantized,'
                             ' while received type %s' % str(type(excluded_sym_names)))

        logger.info('Quantizing graph')
        if quantized_dtype not in ('int8', 'uint8', 'auto'):
            raise ValueError('unknown quantized_dtype %s received,'
                             ' expected `int8`, `uint8` or `auto`' % quantized_dtype)
        qsym = _quantize_symbol(sym, excluded_symbols=excluded_sym_names,
                                offline_params=list(arg_params.keys()),
                                quantized_dtype=quantized_dtype)

        th_dict = {}
        collector = None
        if calib_mode is not None and calib_mode != 'none':
            if calib_mode == 'entropy':
                collector = _LayerOutputCollector(
                    include_layer=calib_layer, logger=logger)
                logger.info(
                    'Create a layer output collector for entropy calibration.')
            elif calib_mode == 'naive':
                collector = _LayerOutputMinMaxCollector(
                    include_layer=calib_layer, logger=logger)
                logger.info(
                    'Create a layer output minmax collector for naive calibration')
            else:
                raise ValueError('unknown calibration mode %s received,'
                                 ' expected `none`, `naive`, or `entropy`' % calib_mode)
            logger.info('Collector created, please use set_monitor_callback'
                        ' to collect calibration information.')

        logger.info('Quantizing parameters')
        qarg_params = _quantize_params(qsym, arg_params, th_dict)

        return qsym, qarg_params, aux_params, collector

    def calib_graph(qsym, arg_params, aux_params, collector,
                    calib_mode='entropy', quantized_dtype='int8', logger=logging):

        th_dict = {}
        if calib_mode is not None and calib_mode != 'none':
            if calib_mode == 'entropy':
                logger.info('Calculating optimal thresholds for quantization')
                th_dict = _get_optimal_thresholds(
                    collector.nd_dict, quantized_dtype, logger=logger)
            elif calib_mode == 'naive':
                th_dict = collector.min_max_dict
            else:
                raise ValueError('unknown calibration mode %s received,'
                                 ' expected `none`, `naive`, or `entropy`' % calib_mode)
            logger.info('Calibrating quantized symbol')
            qsym = _calibrate_quantized_sym(qsym, th_dict)
        else:
            raise ValueError('please set calibration mode to naive or entropy.')

        logger.info('Quantizing parameters')
        qarg_params = _quantize_params(qsym, arg_params, th_dict)

        return qsym, qarg_params, aux_params

    mod = mx.mod.Module.load('model_wo_ringbuffer', 0)
    mod.bind(for_training=False, data_shapes=[('data', (1, 1, 76, 64))], label_shapes=None)
    sym = mod.symbol
    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')
    arg_params, aux_params = mod.get_params()
    # quantize configs
    # set exclude layers
    excluded_names = ['hybridsequential0_conv0_fwd']
    # set calib mode.
    calib_mode = 'none'
    # set calib_layer
    calib_layer = None
    # set quantized_dtype
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    quantized_dtype = 'uint8'
    logger.info('Quantizing FP32 model wakeword')
    qsym, qarg_params, aux_params, collector = quantize_graph(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                              excluded_sym_names=excluded_names,
                                                              calib_mode=calib_mode, calib_layer=calib_layer,
                                                              quantized_dtype=quantized_dtype, logger=logger)
    mx.model.save_checkpoint('quantized_model_wo_ringbuffer_wo_calibration', 0, qsym, qarg_params, aux_params)

    num_examples = 100
    batch_size = 20
    calib_data = np.random.rand(num_examples, 1, 76, 64)
    calib_label = np.random.randint(0, 1, size=(num_examples, ))
    calibrate_iter = mx.io.NDArrayIter(calib_data, calib_label, batch_size, shuffle=True)
    # set calib mode.
    calib_mode = 'naive'
    # set calib_layer
    calib_layer = None
    logger.info('Quantizing FP32 model ww')
    cqsym, cqarg_params, aux_params, collector = quantize_graph(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                                excluded_sym_names=excluded_names,
                                                                calib_mode=calib_mode, calib_layer=calib_layer,
                                                                quantized_dtype=quantized_dtype, logger=logger)

    # create module
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=calibrate_iter.provide_data, label_shapes=None)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # calibration configs
    # set num_calib_batches
    num_calib_batches = 5
    max_num_examples = num_calib_batches * batch_size

    # monitor FP32 Inference
    mod._exec_group.execs[0].set_monitor_callback(collector.collect, monitor_all=True)
    num_batches = 0
    num_examples = 0
    for batch in calibrate_iter:
        mod.forward(data_batch=batch, is_train=False)
        num_batches += 1
        num_examples += batch_size
        if num_examples >= max_num_examples:
            break
    if logger is not None:
        logger.info("Collected statistics from %d batches with batch_size=%d"
                    % (num_batches, batch_size))

    # write scaling factor into quantized symbol
    cqsym, cqarg_params, aux_params = calib_graph(qsym=cqsym, arg_params=arg_params, aux_params=aux_params,
                                                  collector=collector, calib_mode=calib_mode,
                                                  quantized_dtype=quantized_dtype, logger=logger)
    cqsym = cqsym.get_backend_symbol('MKLDNN_QUANTIZE')
    mx.model.save_checkpoint('quantized_model_wo_ringbuffer_w_fusion', 0, cqsym, cqarg_params, aux_params)


from mxnet.contrib.quantization import *
def quantize_ww_model_with_activation_fused():
    import mxnet as mx
    import logging

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    mod = mx.mod.Module.load('model_wo_ringbuffer', 0)
    mod.bind(for_training=False, data_shapes=[('data', (1, 1, 76, 64))], label_shapes=None)
    sym = mod.symbol
    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')
    arg_params, aux_params = mod.get_params()
    # quantize configs
    # set exclude layers
    # excluded_names = ['hybridsequential0_conv0_fwd']
    excluded_names = []
    # set calib mode.
    calib_mode = 'none'
    # set calib_layer
    calib_layer = None
    # set quantized_dtype
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    quantized_dtype = 'uint8'
    logger.info('Quantizing FP32 model wakeword')
    qsym, qarg_params, aux_params, collector = quantize_graph(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                              excluded_sym_names=excluded_names,
                                                              calib_mode=calib_mode, calib_layer=calib_layer,
                                                              quantized_dtype=quantized_dtype, logger=logger)
    mx.model.save_checkpoint('quantized_model_wo_ringbuffer_wo_calibration', 0, qsym, qarg_params, aux_params)

    num_examples = 100
    batch_size = 20
    calib_data = np.random.rand(num_examples, 1, 76, 64)
    calib_label = np.random.randint(0, 1, size=(num_examples, ))
    calibrate_iter = mx.io.NDArrayIter(calib_data, calib_label, batch_size, shuffle=True)
    # set calib mode.
    calib_mode = 'naive'
    # set calib_layer
    calib_layer = None
    logger.info('Quantizing FP32 model ww')
    cqsym, cqarg_params, aux_params, collector = quantize_graph(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                                excluded_sym_names=excluded_names,
                                                                calib_mode=calib_mode, calib_layer=calib_layer,
                                                                quantized_dtype=quantized_dtype, logger=logger)

    # create module
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=calibrate_iter.provide_data, label_shapes=None)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # calibration configs
    # set num_calib_batches
    num_calib_batches = 5
    max_num_examples = num_calib_batches * batch_size

    # monitor FP32 Inference
    mod._exec_group.execs[0].set_monitor_callback(collector.collect, monitor_all=True)
    num_batches = 0
    num_examples = 0
    for batch in calibrate_iter:
        mod.forward(data_batch=batch, is_train=False)
        num_batches += 1
        num_examples += batch_size
        if num_examples >= max_num_examples:
            break
    if logger is not None:
        logger.info("Collected statistics from %d batches with batch_size=%d"
                    % (num_batches, batch_size))

    # write scaling factor into quantized symbol
    cqsym, cqarg_params, aux_params = calib_graph(qsym=cqsym, arg_params=arg_params, aux_params=aux_params,
                                                  collector=collector, calib_mode=calib_mode,
                                                  quantized_dtype=quantized_dtype, logger=logger)
    cqsym = cqsym.get_backend_symbol('MKLDNN_QUANTIZE')
    mx.model.save_checkpoint('quantized_model_wo_ringbuffer_w_fusion', 0, cqsym, cqarg_params, aux_params)





if __name__ == '__main__':
    # test_qnn()
    # test_conv_with_bias()
    # test_conv_with_bias()
    # test_quantized_wakeword_model()
    run_intermediate_quantized_layers()
    # run_intermediate_quantized_layers_uint8()
    # cross_compile_for_echo_dot2()
    # cross_compile_for_fp32_model_echo_dot2()
    # ww_fp32_model_local()
    # ww_int8_model_local_accuracy_validation()
    # ww_int8_mxnet_model_local_accuracy_validation()
    # ww_fp32_mxnet_model_local_accuracy_validation()