/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file quantize_rewrite.cc
 * \brief Lower quantized ops to exisiting Relay ops.
 */

#include <tvm/relay/pass.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/nn_quantize.h>
#include "pattern_util.h"

namespace tvm {
    namespace relay {

        Expr ConvolveQuantizedTensors(const Expr &quantized_data,
                                      const Expr &quantized_kernel, const QuantizedConv2DAttrs *&param) {
            // TODO (janimesh) - Who should decide the accumulation dtype?
            if (param->input_zero_point == 0 && param->kernel_zero_point == 0) {
                Expr int8_conv = Conv2D(quantized_data,
                                        quantized_kernel,
                                        param->strides,
                                        param->padding,
                                        param->dilation,
                                        param->groups,
                                        param->channels,
                                        param->kernel_size,
                                        param->data_layout,
                                        param->kernel_layout,
                                        param->out_layout,
                                        Int(32));
                return int8_conv;
            }
            LOG(FATAL) << "Only symmetric quantization supported";
            return Expr(); // to hide the warning.
        }

        Expr ScaleHandling(const Expr &convolved_tensor,
                           const QuantizedConv2DAttrs *&param) {
            // The scale handling can be done in many ways.
            // 1) Floating point handling
            //    Here we can multiply the scale to the convolved_tensor, round to nearest
            //    integer and then cast back to int32.
            // 2) Integer only scale handling
            //    Here, the computation is converted to a fixed point computation by
            //    computing output multiplier and shift. This is useful, if the target
            //    device does not support/have very expensive floating point computations.

            if (param->use_integer_computation_for_scale_handling == false) {
                double multiplier = (param->input_scale * param->kernel_scale) /
                                    param->output_scale;
                auto scalar_multiplier = MakeConstantScalar(Float(32), multiplier);
                auto casted_convolved_tensor = Cast(convolved_tensor, Float(32));
                auto scaled_fp32_tensor = Multiply(casted_convolved_tensor, scalar_multiplier);
                auto scaled_rounded_fp32_tensor = Round(scaled_fp32_tensor);
                auto scaled_tensor = Cast(scaled_rounded_fp32_tensor, Int(32));
                return scaled_tensor;
            }
            LOG(FATAL) << "Only floating point scale handling is supported for now.";
            return Expr(); // to hide the warning.
        }

        Expr ReQuantize(const Expr &scaled_output,
                        const QuantizedConv2DAttrs *&param) {
            Expr requantized_output = Cast(scaled_output, param->out_dtype);
            return requantized_output;
        }

        Expr QuantizedConv2DForwardRewrite(const Call &ref_call,
                                           const Array<Expr> &new_args,
                                           const NodeRef &ctx) {
            CHECK_EQ(new_args.size(), 2);
            Expr quantized_data = new_args[0];
            Expr quantized_kernel = new_args[1];
            const auto *param = ref_call->attrs.as<QuantizedConv2DAttrs>();

            // Check for current quantization support.
            CHECK_EQ(param->input_zero_point, 0)
                << "Encountered non-zero zero point."
                << " Only symmetric quantization supported for now.";
            CHECK_EQ(param->kernel_zero_point, 0)
                << "Encountered non-zero zero point."
                << " Only symmetric quantization supported for now.";
            CHECK_EQ(param->output_zero_point, 0)
                << "Encountered non-zero zero point."
                << " Only symmetric quantization supported for now.";
            CHECK_EQ(param->use_integer_computation_for_scale_handling, false)
                << "Currently floating point computation is used for scale handling. "
                << "Please switch to False if HW supports floating point arithmetic";

            // Lowering of the quantized_convolution.
            //
            // For FP32, the conv output is
            //     C = conv(A, W)
            // or, C(n, oc, oh, ow) = A(n, ic, oh + r, ow + s) * W(oc, ic, r, s)
            // where, ic, r, s are reduce axis.
            //
            // For quantized convolution, each tensor is represented in quantized format
            //    A = scale_a x (QA - zp_A)
            // where QA is quantized tensor, scale_a and zp_A are quantizations params.
            //
            // For symmetric quantization, the zp_* for all tensors is 0.
            // So, the quantized_convolution becomes
            //
            //    scale_c * QC(n, oc, oh, ow) =
            //        scale_a * QA(n, ic, oh + r, ow + s) x
            //        scale_w * QW(oc, ic, r, s)
            //
            // So, to get the quantized tensor C, the computation is
            //
            //    QC(n, oc, oh, ow) = (scale_a * scale_w)/scale_c x
            //        QA(n, ic, oh + r, ow + s) x QW(oc, ic, r, s)
            //
            // or,
            //    QC = K * conv(QA, QB)
            //
            // For asymmetric computation, we can perform similar unrolling. We can find
            // more details at
            // https://discuss.tvm.ai/t/tf-lite-quantized-conv2d-operator-conversion/2651/8?u=janimesh

            // The above computation is arranged in following functions
            //    1) ConvolveQuantizedTensors
            //        a) For symmetric, conv(QA, QB).
            //        b) For asymmetric, it involves 4 terms.
            //    2) ScaleHandling
            //        a) Takes convolved output and scales it.
            //        b) Can support both float and integer computation.
            //    3) Requantize
            //        a) Converts the intermediate dtype back to int8.
            Expr convolved_tensor = ConvolveQuantizedTensors(quantized_data,
                                                             quantized_kernel,
                                                             param);
            Expr scaled_output = ScaleHandling(convolved_tensor, param);
            Expr requantized_output = ReQuantize(scaled_output, param);
            // TODO(janimesh) - Look at the literature and use the right scale
            // calculations.
            return requantized_output;
        }

        RELAY_REGISTER_OP("nn_quantized.quantized_conv2d")
        .set_attr<FForwardRewrite>("FQuantizeForwardRewrite", QuantizedConv2DForwardRewrite);

        TVM_REGISTER_API("relay._quantize.quantize_rewrite")
        .set_body_typed<Expr(Expr)>([](const Expr &e) {
            Expr ret = ForwardRewrite(e, "FQuantizeForwardRewrite", nullptr, nullptr);
            return ret;
        });


        /* quantized relu */
        Expr ReluQuantizedTensors(const Expr &quantized_data, const QuantizedReluAttrs *&param) {
            if (param->input_zero_point == 0) {
                Expr int8_relu = Relu(quantized_data);
                return int8_relu;
            }
            LOG(FATAL) << "Only symmetric quantization supported";
            return Expr(); // to hide the warning.
        }

        Expr ReQuantize(const Expr &relued_output,
                        const QuantizedReluAttrs *&param) {
            Expr requantized_output = Cast(relued_output, param->out_dtype);
            return requantized_output;
        }

        Expr QuantizedReluForwardRewrite(const Call &ref_call,
                                         const Array<Expr> &new_args,
                                         const NodeRef &ctx) {
            CHECK_EQ(new_args.size(), 1);
            Expr quantized_data = new_args[0];
            const auto *param = ref_call->attrs.as<QuantizedReluAttrs>();

            // Check for current quantization support.
            CHECK_EQ(param->input_zero_point, 0)
                << "Encountered non-zero zero point."
                << " Only symmetric quantization supported for now.";
            CHECK_EQ(param->output_zero_point, 0)
                << "Encountered non-zero zero point."
                << " Only symmetric quantization supported for now.";

            Expr relued_tensor = ReluQuantizedTensors(quantized_data, param);
            Expr requantized_output = ReQuantize(relued_tensor, param);
            return relued_tensor;
        }

        RELAY_REGISTER_OP("nn_quantized.quantized_relu")
        .set_attr<FForwardRewrite>("FQuantizeForwardRewrite", QuantizedReluForwardRewrite);

    }  // namespace relay
}  // namespace tvm
