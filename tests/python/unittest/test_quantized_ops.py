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
from tvm import relay
from tvm.relay.testing import create_workload

def test_quantized_conv2d():
    quantized_data = relay.var("quantized_data", shape=(1, 128, 16, 16), dtype='int8')
    quantized_weight = relay.var("weight", shape=(64, 128, 3, 3), dtype='int8')
    quantized_output = relay.op.nn._quantize.quantized_conv2d( \
        quantized_data, quantized_weight,
        input_zero_point=0,
        kernel_zero_point=0,
        output_zero_point=0,
        input_scale=0.5,
        kernel_scale=0.5,
        output_scale=0.5,
        channels=64,
        kernel_size=(3,3),
        out_dtype="int8")
    func = relay.Function(relay.ir_pass.free_vars(quantized_output),
                          quantized_output)
    print("###### Original graph starts ######")
    print(func)
    print("###### Original graph ends ######")
    func = relay.ir_pass.infer_type(func)
    print("###### TypeInferred graph starts ######")
    print(func)
    print("###### TypeInferred graph ends ######")
    func = relay.quantize.quantize_rewrite(func)
    func = relay.ir_pass.infer_type(func)
    print("###### Lowered graph starts ######")
    print(func)
    print("###### Lowered graph ends ######")

if __name__ == "__main__":
    test_quantized_conv2d()