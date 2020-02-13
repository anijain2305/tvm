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
from tvm.ir import IRModule
from tvm import relay
from tvm.relay.transform import FastMath

def test_exp():
    x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
    y = relay.exp(x)
    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)

    fast_mod = FastMath()(mod)
    assert "fast_exp" in fast_mod.astext()

    # Check that opt level 4 triggers the transformation.
    with relay.build_config(opt_level=4):
        fast_mod = relay.optimize(mod, target='llvm', params=None)
    assert "fast_exp" in fast_mod[0].astext()

def test_tanh():
    x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
    y = relay.tanh(x)
    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)

    fast_mod = FastMath()(mod)
    assert "fast_tanh" in fast_mod.astext()

    # Check that opt level 4 triggers the transformation.
    with relay.build_config(opt_level=4):
        fast_mod = relay.optimize(mod, target='llvm', params=None)
    assert "fast_tanh" in fast_mod[0].astext()

if __name__ == "__main__":
    test_exp()
    test_tanh()
