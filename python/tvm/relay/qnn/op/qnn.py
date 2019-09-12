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
#pylint: disable=invalid-name
"""QNN dialect operators."""

from __future__ import absolute_import as _abs
from tvm.expr import FloatImm, IntImm
from tvm.relay.expr import Tuple
import numpy as np
from . import _make

zero_centered_uint8_quantized_range = np.float32(255)
zero_centered_int8_quantized_range = np.float32(127)

def requantize(data,
               input_scale,
               input_zero_point,
               output_scale,
               output_zero_point,
               rounding="TONEAREST",
               out_dtype="int8"):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: float
        The quantization scale for the input tensor.

    input_zero_point: int
        The zero point of the input tensor.

    output_scale: float
        The quantization scale for the output tensor.

    output_zero_point: int
        The zero point of the output tensor.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    out_dtype : str, optional
        Specifies the output data type.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.requantize(data,
                            input_scale,
                            input_zero_point,
                            output_scale,
                            output_zero_point,
                            rounding,
                            out_dtype)


def quantize(data,
             output_scale,
             output_zero_point,
             out_dtype='int8'):
    r""" Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : int
        The output zero_point.
    output_scale : float
        The output scale.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.quantize(data,
                          output_scale,
                          output_zero_point,
                          out_dtype)


def _quantize_with_zero_centered(data,
                                 data_min,
                                 data_max,
                                 quantized_range,
                                 out_dtype):
    r"""Quantizes the given data tensor by calculating the scale
    using the MKLDNN formula `quantized_range / max(abs(data_min, data_max))`.
    Where quantized_range is 255 for uint8 and 127 for int8. The `data_min`
    and `data_max` are the min and max to use for the `data` tensor elements.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    data_min : float
        The minimum to use data elements.
    data_max : float
        The maximum to use for data elements.
    quantized_range : float
        255 for uint8 and 127 for int8. This is the data type range.
    out_dtype : str
        The output data type. Can be int8 or uint8
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    real_range = np.max([np.abs(np.float32(data_min)),
                         np.abs(np.float32(data_max))])
    scale = np.divide(quantized_range, real_range)
    scale_inverse = np.divide(1.0, scale)
    zero_point = 0
    return quantize(data,
                    scale_inverse,
                    zero_point,
                    out_dtype=out_dtype)


def _quantize_mxnet_min_max_uint8(data,
                                  imin_range,
                                  imax_range):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `uint8`.
    The method of quantizing is described here - https://tinyurl.com/y4d7hrzf.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, Mxnet
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    iinfo = np.iinfo(np.uint8)
    min_limit = np.float64(iinfo.min)
    max_limit = np.float64(iinfo.max)
    imin_range = np.float64(imin_range)
    imax_range = np.float64(imax_range)
    scale = np.divide((max_limit - min_limit),
                      (imax_range - imin_range))
    scale_inverse = np.divide(1.0, scale)
    zero_point = np.int(-1 * imin_range * scale)
    return quantize(data,
                    scale_inverse,
                    zero_point,
                    out_dtype='uint8')


def _quantize_mxnet_min_max_int8(data,
                                 data_min,
                                 data_max):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `int8`.
    The method of quantizing is described here - https://tinyurl.com/y4d7hrzf.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, Mxnet
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _quantize_with_zero_centered(data,
                                        data_min,
                                        data_max,
                                        zero_centered_int8_quantized_range,
                                        'int8')


def _quantize_mkldnn_min_max_uint8(data,
                                   data_min,
                                   data_max):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `uint8`.
    The method of quantizing is described here - https://tinyurl.com/y5k6fz5w.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, MKLDNN
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _quantize_with_zero_centered(data,
                                        data_min,
                                        data_max,
                                        zero_centered_uint8_quantized_range,
                                        'uint8')


def _quantize_mkldnn_min_max_int8(data,
                                  data_min,
                                  data_max):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `int8`.
    The method of quantizing is described here - https://tinyurl.com/y5k6fz5w.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, MKLDNN
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _quantize_with_zero_centered(data,
                                        data_min,
                                        data_max,
                                        zero_centered_int8_quantized_range,
                                        'int8')


def quantize_mxnet_min_max(data,
                           min_range,
                           max_range,
                           out_dtype='int8',
                           use_mkldnn=False):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type.
    Only `int8` and `uint8` is supported as output data types.
    The input data type is expected to be `float32`.
    Mxnet has two different flavors for quantization 1) Default 2)MKLDNN.
    To get the second one Mxnet must be built with MKLDNN during compile time.
    Users can choose either of the implementation for TVM runtime.
    The main difference between the two implementation is that MKLDNN is centered
    around 0 and the default implementation for uint8 is not.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.
    out_dtype: str, optional
        The output data type, can be 'int8' or 'uint8'
    use_mkldnn: bool, optional
        If True then uses MKLDNN quantization implementation otherwise
        will use default implementation.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if out_dtype == 'uint8':
        if use_mkldnn:
            return _quantize_mkldnn_min_max_uint8(data,
                                                  min_range,
                                                  max_range)
        else:
            return _quantize_mxnet_min_max_uint8(data,
                                                 min_range,
                                                 max_range)
    elif out_dtype == 'int8':
        if use_mkldnn:
            return _quantize_mkldnn_min_max_int8(data,
                                                 min_range,
                                                 max_range)
        else:
            return _quantize_mxnet_min_max_int8(data,
                                                min_range,
                                                max_range)
    else:
        raise ValueError(
            "Expected out_dtype to be int8 or uint8 but was  %s" % out_dtype)


def dequantize(data,
               input_scale,
               input_zero_point):
    r""" Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point : int
        The output zero_point.
    input_scale : float
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dequantize(data,
                            input_scale,
                            input_zero_point)


def concatenate(data,
                input_scales,
                input_zero_points,
                output_scale,
                output_zero_point,
                axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        The list of quantized tensors.

    input_scales : List[float32]
        The list of scales of input quantized tensors.

    input_zero_points : List[int32]
        The list of zero points of input quantized tensors.

    output_scale : float32
        The scale of the output quantized tensor.

    output_zero_point : int32
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")

    return _make.concatenate(Tuple(data),
                             [FloatImm("float64", x) for x in input_scales],
                             [IntImm("int32", x) for x in input_zero_points],
                             output_scale,
                             output_zero_point,
                             axis)


def conv2d(data,
           kernel,
           input_zero_point,
           kernel_zero_point,
           strides=(1, 1),
           padding=(0, 0),
           dilation=(1, 1),
           groups=1,
           channels=None,
           kernel_size=None,
           data_layout="NCHW",
           kernel_layout="OIHW",
           out_layout="",
           out_dtype="int32"):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.conv2d(data, kernel,
                        input_zero_point, kernel_zero_point,
                        strides, padding, dilation,
                        groups, channels, kernel_size,
                        data_layout, kernel_layout, out_layout, out_dtype)


def add(lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale,
        output_zero_point):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.add(lhs, rhs,
                     lhs_scale, lhs_zero_point,
                     rhs_scale, rhs_zero_point,
                     output_scale, output_zero_point)
