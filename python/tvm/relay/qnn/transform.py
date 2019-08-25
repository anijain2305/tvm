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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
QNN pass transformation infrastructure.
"""
from tvm import relay

def QnnToRelay():
    """Lowers the QNN ops to a sequence of Relay ops. The lowering sequence is defined for each op
    by using the Legalize API. The legalize map attr name is FTVMQnnToRelay.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that lowers QNN to Relay ops.
    """

    return relay.transform.Legalize("FTVMQnnToRelay")


def Legalize():
    """Legalizes only QNN ops to another sequence of QNN/Relay ops. The legalization can be
    configured to happen per target. The legalize map attr name is FTVMQnnLegalize.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that legalizes QNN ops.
    """

    return relay.transform.Legalize("FTVMQnnLegalize")
