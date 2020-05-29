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
"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np
import glob

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt


# from compress import ModelCompressor
from src.model_compressor import ModelCompressor

def preprocessing(image_instance, image_shape):
    image = Image.open(image_instance).resize(image_shape)
    image = np.array(image)
    if len(image.shape) == 2:
        # replicate the third channel
        image = np.reshape(image, newshape=(image_shape[0], image_shape[1], 1))
        image = np.repeat(image, 3, axis=2)

    image = image - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def run_imagenet(m, image_shape):
    image_path = '/home/ubuntu/imagenet/val/'
    all_class_path = sorted(glob.glob(image_path+'*'))
    total = 0
    top1_score = 0
    top5_score = 0
    label = 0
    num_images = 0

    tflite_top1 = 0
    tflite_top5 = 0

    for cur_class in all_class_path:
        all_image = glob.glob(cur_class+'/*')
        for image_instance in all_image:
            total = total + 1
            num_images = num_images + 1

            # Preprocess the image
            preprocessed_image = preprocessing(image_instance, image_shape)

            m.set_input("data", preprocessed_image)

            # Set the new inputs
            m.run()

            # PostProcess
            predictions = m.get_output(0).asnumpy().squeeze()

            # Get the labels
            labels_sorted = predictions.argsort()[-5:][::-1]

            gt = label
            # Collect statistics
            if labels_sorted[0] == gt:
                top1_score = top1_score + 1
            if gt in labels_sorted:
                top5_score = top5_score + 1

            if (num_images == 1000):
                return (top1_score/num_images, top5_score/num_images)
                # print("Results", str(top1_score/num_images), str(top5_score/num_images), sep="\t")
                # return
        label = label + 1
    return (top1_score/num_images, top5_score/num_images)
    # print("Results", str(top1_score/num_images), str(top5_score/num_images), sep="\t")

def compile_run(mod, params, image_shape):
    ## we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    ######################################################################
    # now compile the graph
    # target = 'llvm -mcpu=cascadelake'
    target = 'llvm -mcpu=skylake'
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now, we would like to reproduce the same forward computation using TVM.
    from tvm.contrib import graph_runtime
    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(**params)

    # Run Imagenet
    return run_imagenet(m, image_shape)


# for model in ["resnet50_v1"]:
models = ["resnet18_v1", "resnet50_v1", "inceptionv3", "mobilenet1.0", "mobilenetv2_1.0"]
models = ["resnet18_v1", "resnet50_v1", "inceptionv3"]

model_shapes = dict()
for model in models:
    model_shapes[model] = (224, 224)

model_shapes["inceptionv3"] = (299, 299)


for model in models:
    block = get_model(model, pretrained=True)
    image_shape = model_shapes[model]
    shape_dict = {'data': (1, 3, image_shape[0], image_shape[1])}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)

    mc = ModelCompressor()
    mc.compress(params, mod['main'], None, "no_decomp")
    compressed_params = mc._optimized_params
    (top1, top5) = compile_run(mod, compressed_params, image_shape)
    print("Result", model, 1.0, "original", top1, top5, mc._total_flops, mc._total_memory,
            mc._l2_norm, "no_skip", sep=",")

    ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0]

    for ratio in ratios:
        for method in ["weight_svd", "spatial_svd", "tucker_decomp", "tensor_train_decomp"]:
            mc = ModelCompressor()
            mc.compress(params, mod['main'], ratio, method)
            compressed_params = mc._optimized_params

            (top1, top5) = compile_run(mod, compressed_params, image_shape)
            print("Result", model, ratio, method, top1, top5, mc._total_flops, mc._total_memory,
                    mc._l2_norm, "no_skip", sep=",")

            if mc._first_conv is not None:
                assert mc._first_conv in compressed_params
                compressed_params[mc._first_conv] = params[mc._first_conv]

                compressed_first_conv_stats = mc._stats[mc._first_conv]
                original_first_conv_staus = mc._first_conv_orig_stats[mc._first_conv]

                total_flops = mc._total_flops \
                              - compressed_first_conv_stats[0] \
                              + original_first_conv_staus[0]
                total_memory = mc._total_memory \
                               - compressed_first_conv_stats[1] \
                               + original_first_conv_staus[1]
                total_l2_norm = mc._l2_norm \
                               - compressed_first_conv_stats[2] \
                               + original_first_conv_staus[2]
                (top1, top5) = compile_run(mod, compressed_params, image_shape)
                print("Result", model, ratio, method, top1, top5, total_flops, total_memory,
                       total_l2_norm, "first_conv_skip", sep=",")
