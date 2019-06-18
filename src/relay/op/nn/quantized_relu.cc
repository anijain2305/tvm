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
 * \file quantized_convolution.cc
 * \brief Quantized convolution operators
 */

#include <tvm/data_layout.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/attrs/nn_quantize.h>

namespace tvm {
    namespace relay {

        TVM_REGISTER_NODE_TYPE(QuantizedReluAttrs);

        bool QuantizeReluRel(const Array<Type>& types,
                               int num_inputs,
                               const Attrs& attrs,
                               const TypeReporter& reporter) {
            CHECK_EQ(types.size(), 2);
            const auto* data = types[0].as<TensorTypeNode>();
            if (data == nullptr) return false;

            const QuantizedReluAttrs* param = attrs.as<QuantizedReluAttrs>();
            CHECK(param != nullptr);
            DataType out_dtype = data->dtype;
            CHECK_NE(out_dtype, NullValue<DataType>())
                << "Quantized convolution out_dtype has to be passed\n";

            Array<IndexExpr> dshape = data->shape;
            Array<IndexExpr> oshape = dshape;
            // assign output type
            reporter->Assign(types[1], TensorTypeNode::make(oshape, out_dtype));
            return true;
        }


// Positional relay function to create quantized relu operator
// used by frontend FFI.
        Expr MakeQuantizeRelu(Expr quantized_data,
                              int32_t input_zero_point,
                              int32_t output_zero_point,
                              double input_scale,
                              double output_scale,
                              DataType out_dtype) {
            auto attrs = make_node<QuantizedReluAttrs>();
            attrs->out_dtype = std::move(out_dtype);
            attrs->input_zero_point = std::move(input_zero_point);
            attrs->output_zero_point = std::move(output_zero_point);
            attrs->input_scale = std::move(input_scale);
            attrs->output_scale = std::move(output_scale);
            static const Op& op = Op::Get("nn_quantized.quantized_relu");
            return CallNode::make(op, {quantized_data}, Attrs(attrs), {});
        }

        RELAY_REGISTER_OP("nn_quantized.quantized_relu")
                .describe(R"code(2D quantized relu layer.

Returns the relu input array, computed element-wise.

.. math::
   max(x, 0)

- **quantized_data**: Input is any shape of array
- **quantized_out**:  Output has same data shape with input

)code" TVM_ADD_FILELINE)
                .set_attrs_type_key("relay.attrs.QuantizedReluAttrs")
                .set_num_inputs(1)
                .add_argument("quantized_data", "Tensor", "The quantized input quantized_data tensor.")
                .set_support_level(1)
                .add_type_rel("QuantizeRelu", QuantizeReluRel);

        TVM_REGISTER_API("relay.op.nn._quantize._make.quantized_relu")
        .set_body_typed(MakeQuantizeRelu);

    }  // namespace relay
}  // namespace tvm
