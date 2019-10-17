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
 *  Copyright (c) 2019 by Contributors
 * \file src/relay/qnn/op/dense.cc
 * \brief Property def of qnn dense operator.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../op/nn/nn.h"
#include "../../pass/pattern_util.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.dense
TVM_REGISTER_NODE_TYPE(QnnDenseAttrs);

bool QnnDenseRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;
  const auto* param = attrs.as<QnnDenseAttrs>();
  CHECK(param != nullptr) << "QnnDenseAttrs cannot be nullptr.";
  CHECK(data->dtype == Int(8) || data->dtype == UInt(8))
    << "Expected quantized dense type(int8, uint8) for input but was " <<  data->dtype;
  CHECK(weight->dtype == Int(8) || weight->dtype == UInt(8))
    << "Expected quantized dense type(int8, uint8) for weight but was " <<  weight->dtype;
  CHECK(param->out_dtype == Int(32))
    << "Expected quantized dense type(int32) for output but was " <<  param->out_dtype;
  CHECK(param->out_dtype.bits() > 0) << "Output dtype bits should be greater than 0.";
  return DenseRel<QnnDenseAttrs>(types, num_inputs, attrs, reporter);
}

// Positional relay function to create quantized dense operator used by frontend FFI.
Expr MakeQuantizedDense(Expr data,
                        Expr weight,
                        IndexExpr units,
                        int32_t input_zero_point,
                        int32_t kernel_zero_point,
                        DataType out_dtype) {
  auto attrs = make_node<QnnDenseAttrs>();
  attrs->units = std::move(units);
  attrs->out_dtype = out_dtype;
  attrs->input_zero_point = input_zero_point;
  attrs->kernel_zero_point = kernel_zero_point;
  static const Op& op = Op::Get("qnn.dense");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

Expr DenseFirstTerm(const Expr& quantized_data,
                    const Expr& quantized_kernel,
                    const QnnDenseAttrs* attrs) {
  return Dense(quantized_data, quantized_kernel, attrs->units,  attrs->out_dtype);
}

Expr DenseSecondTerm(const Expr& quantized_data,
                     const Expr& zp_kernel) {
  Array<Integer> axes = {1};
  return Multiply(zp_kernel, Sum(Cast(quantized_data, Int(32)), axes, true, false));
}

Expr DenseThirdTerm(const Expr& quantized_kernel,
                    const Expr& zp_data) {
  Array<Integer> axes = {1};
  return Multiply(zp_data, Sum(Cast(quantized_kernel, Int(32)), axes, false, false));
}

Expr DenseFourthTerm(const QnnDenseAttrs* attrs,  int common_axes) {
  int64_t scalar_term = attrs->input_zero_point * attrs->kernel_zero_point * common_axes;
  return MakeConstantScalar(Int(32), (int32_t)scalar_term);
}

Expr QnnDenseCanonicalize(const Attrs& attrs,
                          const Array<Expr>& new_args,
                          const Array<tvm::relay::Type>& arg_types) {
  CHECK_EQ(new_args.size(), 2);
  Expr quantized_data = new_args[0];
  Expr quantized_kernel = new_args[1];
  const auto* qnn_dense_attrs = attrs.as<QnnDenseAttrs>();
  auto term1 = DenseFirstTerm(quantized_data, quantized_kernel, qnn_dense_attrs);
  auto zp_kernel = MakeConstantScalar(Int(32), qnn_dense_attrs->kernel_zero_point);
  auto term2 = DenseSecondTerm(quantized_data, zp_kernel);
  auto zp_data = MakeConstantScalar(Int(32), qnn_dense_attrs->input_zero_point);
  auto term3 = DenseThirdTerm(quantized_kernel, zp_data);
  auto get_shape = [](const Type& type) {
    auto input_tt = type.as<TensorTypeNode>();
    CHECK(input_tt != nullptr) << "Type information missing."
                               << " Please run infer_type pass.";
    return input_tt->shape;
  };
  const auto in_shape = get_shape(arg_types[0]);
  auto term4 = DenseFourthTerm(qnn_dense_attrs, get_const_int(in_shape[0]));
  if (qnn_dense_attrs->input_zero_point == 0 && qnn_dense_attrs->kernel_zero_point == 0) {
    // term 2, 3 and 4 become zero.
    return term1;
  } else if (qnn_dense_attrs->input_zero_point == 0 && qnn_dense_attrs->kernel_zero_point != 0) {
    // term 3 and term 4 become zero.
    return Subtract(term1, term2);
  } else if (qnn_dense_attrs->input_zero_point != 0 && qnn_dense_attrs->kernel_zero_point == 0) {
    // term 2 and term 4 become zero.
    return Subtract(term1, term3);
  } else {
    auto data_term = Subtract(term1, term2);
    // Putting constant terms together, so that constant folding can fold it.
    auto const_term = Subtract(term4, term3);
    return Add(data_term, const_term);
  }
//  Expr quantized_data_int32 = Cast(quantized_data, Int(32));
//  if (qnn_dense_attrs->input_zero_point != 0) {
//    quantized_data_int32 = Subtract(quantized_data_int32,
//                                    MakeConstantScalar(Int(32),
//                                    qnn_dense_attrs->input_zero_point));
//  }
//  Expr quantized_kernel_int32 = Cast(quantized_kernel, Int(32));
//  if (qnn_dense_attrs->kernel_zero_point != 0) {
//    quantized_kernel_int32 = Subtract(quantized_kernel_int32,
//                                      MakeConstantScalar(Int(32),
//                                      qnn_dense_attrs->kernel_zero_point));
//  }
//  Expr int32_dense = Dense(quantized_data_int32,
//                           quantized_kernel_int32,
//                           qnn_dense_attrs->units,
//                           qnn_dense_attrs->out_dtype);
//  return int32_dense;
}

RELAY_REGISTER_OP("qnn.dense")
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.
- **data**: quantized(int8, unit8) `(x1, x2, ..., xn, input_dim)`
- **weight**: quantized(int8, unit8) `(units, input_dim)`
- **out**: quantized(int32) `(x1, x2, ..., xn, units)`.
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.QnnDenseAttrs")
.set_num_inputs(2)
.add_argument("data", "quantized nD Tensor", "Input data.")
.add_argument("weight", "quantized 2D Tensor", "Weight matrix.")
.set_support_level(11)
.add_type_rel("QDense", DenseRel<QnnDenseAttrs>)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnDenseCanonicalize);

TVM_REGISTER_API("relay.qnn.op._make.dense")
.set_body_typed(MakeQuantizedDense);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
