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
from pathlib import Path

import argparse
import logging
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.contrib.quantization import *
import statistics


target = 'cuda'
target_host = 'llvm -target=aarch64-linux-gnu'
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
set_cuda_target_arch('sm_62')
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


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=False):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                dtype = tasks[i].workload[1][-1]
                if dtype != 'float32':
                    continue
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    if os.path.exists(log_filename):
        os.remove(log_filename)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        n_trial = 500
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)

########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, mod, params, data_shape, out_shape, log_file,
        input_name):
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              target_host=target_host,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # use int8 template_key
    for i in range(len(tasks)):
        tsk = tasks[i]
        if tsk.workload[0] != 'conv2d':
            continue
        dtype = tsk.workload[1][-1]
        if 'int8' not in dtype:
            continue
        input_channel = tsk.workload[2][1]
        output_channel = tsk.workload[2][0]
        if output_channel % 4 == 0 and input_channel % 4 == 0:
            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                      tasks[i].target, tasks[i].target_host, 'int8')
            tasks[i] = tsk

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

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


def compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape):
    tune = True

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
    log_file = "tuned_logs_tx2/" + "%s.log" % model_name

    Path(log_file).touch()

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
        if target == 'cuda':
            tuning_option = {
                'log_filename': log_file,

                'tuner': 'xgb',
                'n_trial': 2000,
                'early_stopping': 600,

                'measure_option': autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=10),
                    # runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                    runner=autotvm.RPCRunner(
                        'janimesh_tx1',  # change the device key to your key
                        '0.0.0.0', 9190,
                        number=10, repeat=1, timeout=10, min_repeat_ms=1500)

                    # runner=autotvm.RPCRunner(
                    #     '1080ti',  # change the device key to your key
                    #     '0.0.0.0', 9190,
                    #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
                ),
			}

        tune_and_evaluate(tuning_option, mod, params, input_shape, out_shape, log_file,
                input_name)

    with autotvm.apply_history_best(log_file):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, target_host=target_host, params=params)
            base = '/home/dlc/janimesh/tvm/mxnet_compiled_models/tvm_' + symbol_file.split('/')[-1].replace('.json','')

            path_lib = base + '_deploy_lib.tar'
            path_graph =  base + '_deploy_graph.json'
            path_params = base + '_deploy_params.params'

            lib.export_library(path_lib)
            with open(path_graph, 'w') as fo:
                fo.write(graph)
            with open(path_params, 'wb') as fo:
                fo.write(relay.save_param_dict(params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
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
    # loading model
    sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)
    compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape)
