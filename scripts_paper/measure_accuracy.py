# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.relay import expr as _expr
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime


import argparse
import logging
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.contrib.quantization import *
import statistics


target = 'llvm -mcpu=cascadelake'
#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.

# You can skip the implementation of this function for this tutorial.
def _bind_params(func, params):
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)



def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc_int8'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        target = 'llvm -mcpu=cascadelake'
        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, input_name, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, mod, params, data_shape, out_shape, log_file, graph_opt_sch_file,
        input_name):
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file, input_name)

    # compile kernels with graph-level best records
    # with autotvm.apply_history_best(log_file):
    # with autotvm.apply_graph_best(graph_opt_sch_file):
    #     with relay.build_config(opt_level=3):
    #         graph, lib, params = relay.build_module.build(
    #             mod, target=target, params=params)

    #     # upload parameters to device
    #     ctx = tvm.cpu()
    #     # data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    #     module = runtime.create(graph, lib, ctx)
    #     # module.set_input(input_name, data_tvm)
    #     module.set_input(**params)

    #     # evaluate
    #     print("Evaluate inference time cost...")
    #     ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
    #     prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    #     print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
    #           (np.mean(prof_res), np.std(prof_res)))


def download_dataset(dataset_url, dataset_dir, logger=None):
    if logger is not None:
        logger.info('Downloading dataset for inference from %s to %s' % (dataset_url, dataset_dir))
    mx.test_utils.download(dataset_url, dataset_dir)


def load_model(symbol_file, param_file, logger=None):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if logger is not None:
        logger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params


def advance_data_iter(data_iter, n):
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False


def score(sym, arg_params, aux_params, data, devs, label_name, max_num_examples, logger=None):
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k=5)]
    if not isinstance(metrics, list):
        metrics = [metrics, ]
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break

    speed = num / (time.time() - tic)

    if logger is not None:
        logger.info('Finished inference with %d images' % num)
        logger.info('Finished with %f images per second', speed)
        logger.warn('Note: GPU performance is expected to be slower than CPU. Please refer quantization/README.md for details')
        for m in metrics:
            logger.info(m.get())


def benchmark_score(symbol_file, ctx, batch_size, num_batches, data_layer_type, logger=None):
    # get mod
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    sym = mx.sym.load(symbol_file_path)
    mod = mx.mod.Module(symbol=sym, context=ctx)
    if data_layer_type == "int8":
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.int8)
    elif data_layer_type == 'uint8':
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.uint8)
    else:  # float32
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.float32)
    mod.bind(for_training=False,
             inputs_need_grad=False,
             data_shapes=[dshape])
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    if data_layer_type == "float32":
        data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx, dtype=data_layer_type)
                for _, shape in mod.data_shapes]
    else:
        data = [mx.nd.full(shape=shape, val=127, ctx=ctx, dtype=data_layer_type)
                for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])  # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)

def compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape):
    tune = False


    input_shape = [1] + list(data_shape)
    input_dict = {'data': input_shape}
    input_name = 'data'

    mod, params = relay.frontend.from_mxnet(sym,
                                            dtype={},
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)
    with tvm.target.create(target):
        mod = relay.qnn.transform.Legalize()(mod)
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        mod = relay.transform.Legalize()(mod)
        expr = _bind_params(mod['main'], params)
        params = None
        mod = relay.Module.from_expr(expr)
        mod = relay.transform.FoldConstant()(mod)

    model_name = symbol_file.split('/')[-1].replace('.json','')
    log_file = "%s.log" % model_name
    graph_opt_sch_file = "%s_graph_opt.log" % model_name
    if tune:
        out_shape = (1, 64, 56, 56)
        tuning_option = {
            'log_filename': log_file,
            'tuner': 'random',
            'early_stopping': None,
        
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1,
                                           min_repeat_ms=1000),
            ),
        }

        tune_and_evaluate(tuning_option, mod, params, input_shape, out_shape, log_file,
                graph_opt_sch_file, input_name)
        
    with autotvm.apply_graph_best(graph_opt_sch_file):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)
            base = '/home/ubuntu/mxnet_compiled_models/tvm_' + symbol_file.split('/')[-1].replace('.json','')
            
            path_lib = base + '_deploy_lib.tar'
            path_graph =  base + '_deploy_graph.json'
            path_params = base + '_deploy_params.params'
            
            lib.export_library(path_lib)
            with open(path_graph, 'w') as fo:
                fo.write(graph)
            with open(path_params, 'wb') as fo:
                fo.write(relay.save_param_dict(params))

 
def profile(data, symbol_file, num_inference_images, sym, devs, label_name):
    debug = False
    import tvm
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime as debug_runtime

    base = '/home/ubuntu/mxnet_compiled_models/tvm_' + symbol_file.split('/')[-1].replace('.json','')

    path_lib = base + '_deploy_lib.tar'
    path_graph =  base + '_deploy_graph.json'
    path_params = base + '_deploy_params.params'

    graph = open(path_graph).read()
    lib = tvm.module.load(path_lib)
    params = bytearray(open(path_params, 'rb').read())

    if debug:
        rt_mod = debug_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.load_params(params)
        rt_mod.run()
        return

    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.load_params(params)

    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data)
#             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    # warm up
    warm_up = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        warm_up += 1
        if warm_up == 50:
            break

    counter = 0
    time_mxnet = list()
    for batch in data:
        time0 = time.time()
        mod.forward(batch, is_train=False)
        nd.waitall() #rt_mod.get_output(0).asnumpy()
        time1 = time.time()
        time_mxnet.append(time1 - time0)
        counter += 1
        if counter == num_inference_images:
            break
    
    # input("MxNet done")
    # # warm up
    # warm_up = 0
    # for batch in data:
    #     rt_mod.run()
    #     warm_up += 1
    #     if warm_up == 50:
    #         break


    # counter = 0
    # time_tvm = list()
    # for batch in data:
    #     rt_mod.set_input('data', batch.data[0].asnumpy())
    #     time0 = time.time()
    #     rt_mod.run()
    #     # nd.waitall() #rt_mod.get_output(0).asnumpy()
    #     time1 = time.time()
    #     time_tvm.append(time1 - time0)
    #     counter += 1
    #     if counter == num_inference_images:
    #         break

    avg = lambda x : round(1000*sum(x)/len(x), 6)
    std = lambda x: round(statistics.stdev(x), 6)


    # total_tvm = avg(time_tvm)
    # sec_tvm = total_tvm/1000
    # std_tvm = std(time_tvm)
    # min_tvm = round(min(time_tvm), 6)
    # deviation_from_min_tvm = round(sec_tvm/min_tvm*100 - 100, 6)
    # deviation_from_std_tvm = round(std_tvm/sec_tvm*100, 6)

    total_mxnet = avg(time_mxnet)
    sec_mxnet = total_mxnet/1000
    std_mxnet = std(time_mxnet)
    min_mxnet = round(min(time_mxnet), 6)
    deviation_from_min_mxnet = round(sec_mxnet/min_mxnet*100 - 100, 6) 
    deviation_from_std_mxnet = round(std_mxnet/sec_mxnet*100, 6)


    # print("TVM time = ", total_tvm, min_tvm, std_tvm, deviation_from_min_tvm, deviation_from_std_tvm)
    print("MXNET time = ", total_mxnet, min_mxnet, std_mxnet, deviation_from_min_mxnet, deviation_from_std_mxnet)
    # print("Speedup = ", total_mxnet/total_tvm)
    # print("Slowdown = ", total_tvm/total_mxnet)

    # if deviation_from_min_tvm > 5:
    #     assert False

def run_tvm(data, symbol_file, num_inference_images, sym, devs, label_name):
    debug = False
    import tvm
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime as debug_runtime

    base = '/home/ubuntu/mxnet_compiled_models/tvm_' + symbol_file.split('/')[-1].replace('.json','')

    path_lib = base + '_deploy_lib.tar'
    path_graph =  base + '_deploy_graph.json'
    path_params = base + '_deploy_params.params'

    graph = open(path_graph).read()
    lib = tvm.module.load(path_lib)
    params = bytearray(open(path_params, 'rb').read())

    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.load_params(params)

    if debug:
        mod = mx.mod.Module(symbol=sym, context=devs) 
        mod.bind(for_training=False,
                 data_shapes=data.provide_data)
    else:
        mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
        mod.bind(for_training=False,
                 data_shapes=data.provide_data,
                 label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    counter = 0
    top_1_raw = 0
    top_5_raw = 0
    top_1_raw_mxnet = 0
    top_5_raw_mxnet = 0
    if debug:
        data = advance_data_iter(data, 0)
    for batch in data:
        # Get the original label.
        correct_label = int(batch.label[0].asnumpy()[0])

        rt_mod.set_input('data', batch.data[0].asnumpy())
        rt_mod.run()
        tvm_res = rt_mod.get_output(0).asnumpy()

        mod.forward(batch, is_train=False)
        mxnet_res = mod.get_outputs()[0].asnumpy()

        if debug:
            print("######## MxNet ###########")
            print(mxnet_res[0][0])
            print(mxnet_res[0][1])
            print(mxnet_res[0][2])
            print("######## TVM ###########")
            print(tvm_res[0][0])
            print(tvm_res[0][1])
            print(tvm_res[0][2])
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("######## MxNet ###########")
            print(mxnet_res)
            print("######## TVM ###########")
            print(tvm_res)
            print("######## Diff ###########")
            # it = np.nditer(mxnet_res, flags=['multi_index'])
            # while not it.finished:
            #     print("%d <%s>" % (it[0], it.multi_index), end='\n')
            #     it.iternext()
            np.testing.assert_allclose(mxnet_res.astype('int32'), tvm_res.astype('int32'), atol=0, verbose=True)
            try:
                np.testing.assert_allclose(mxnet_res.astype('int32'), tvm_res.astype('int32'), atol=0, verbose=True)
            except:
                np.testing.assert_allclose(mxnet_res.astype('int32'), tvm_res.astype('int32'), atol=1, verbose=True)
        else:
            tvm_pred = np.squeeze(tvm_res).argsort()[-5:][::-1]
            mxnet_pred = np.squeeze(mxnet_res).argsort()[-5:][::-1]

            if correct_label == tvm_pred[0]:
                top_1_raw += 1
                top_5_raw += 1
            elif correct_label in tvm_pred:
                top_5_raw += 1


            if correct_label == mxnet_pred[0]:
                top_1_raw_mxnet += 1
                top_5_raw_mxnet += 1
            elif correct_label in mxnet_pred:
                top_5_raw_mxnet += 1

        counter += 1
        if counter == num_inference_images:
            break

    model_name = symbol_file.split('/')[-1].replace('.json','')
    top_1 = float(top_1_raw_mxnet)/float(counter)
    top_5 = float(top_5_raw_mxnet)/float(counter)
    print("Mxnet", model_name, top_1, top_5, sep='\t')


    top_1 = float(top_1_raw)/float(counter)
    top_5 = float(top_5_raw)/float(counter)
    print("Tvm", model_name, top_1, top_5, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--benchmark', type=bool, default=False, help='dummy data benchmark')
    parser.add_argument('--score_tvm', type=bool, default=False, help='score tvm')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--dataset', type=str, required=False, help='dataset path')
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--rgb-std', type=str, default='1,1,1')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--data-layer-type', type=str, default="float32",
                        choices=['float32', 'int8', 'uint8'],
                        help='data type for data layer')

    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    rgb_mean = args.rgb_mean
    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    rgb_std = args.rgb_std
    logger.info('rgb_std = %s' % rgb_std)
    rgb_std = [float(i) for i in rgb_std.split(',')]
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}
    combine_mean_std = {}
    combine_mean_std.update(mean_args)
    combine_mean_std.update(std_args)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    data_layer_type = args.data_layer_type
    if args.benchmark == False:
        dataset = args.dataset
        download_dataset('http://data.mxnet.io/data/val_256_q90.rec', dataset)
        logger.info('Dataset for inference: %s' % dataset)

        # creating data iterator
        data = mx.io.ImageRecordIter(
            path_imgrec=dataset,
            label_width=1,
            preprocess_threads=data_nthreads,
            batch_size=batch_size,
            data_shape=data_shape,
            label_name=label_name,
            rand_crop=False,
            rand_mirror=False,
            shuffle=args.shuffle_dataset,
            shuffle_chunk_seed=args.shuffle_chunk_seed,
            seed=args.shuffle_seed,
            dtype=data_layer_type,
            ctx=args.ctx,
            **combine_mean_std)

        # loading model
        sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)

        # make sure that fp32 inference works on the same images as calibrated quantized model
        logger.info('Skipping the first %d batches' % args.num_skipped_batches)
        data = advance_data_iter(data, args.num_skipped_batches)

        num_inference_images = args.num_inference_batches * batch_size
        logger.info('Running model %s for inference' % symbol_file)
        if args.score_tvm:
            is_profile = False
            # compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape)
            if is_profile:
                profile(data, symbol_file, num_inference_images, sym, [ctx], label_name)
            else:
                run_tvm(data, symbol_file, num_inference_images, sym, [ctx], label_name)
        else:
            score(sym, arg_params, aux_params, data, [ctx], label_name,
                max_num_examples=num_inference_images, logger=logger)
    else:
        logger.info('Running model %s for inference' % symbol_file)
        speed = benchmark_score(symbol_file, ctx, batch_size, args.num_inference_batches, data_layer_type, logger)
        logger.info('batch size %2d, image/sec: %f', batch_size, speed)
