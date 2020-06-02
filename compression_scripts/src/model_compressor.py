import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from topi.util import get_const_tuple

from functools import reduce
import operator

from src.svd_decomposition import WeightSVD, SpatialSVD
from src.no_decomposition import NoDecomposition
from src.cp_decomposition import CPDecomposition
from src.tucker_decomposition import TuckerDecomposition
from src.tensor_train_decomposition import TensorTrainDecomposition
import time

import numpy as np

class ModelCompressor(ExprVisitor):
    def __init__(self):
        super().__init__()
        self._stats = {}

    def compress(self, params, expr, compression_ratio, method, ctx=None):
        self._params = params
        self._optimized_params = dict(params)
        self._compression_ratio = compression_ratio
        self._ctx = ctx
        self._method = method
        self._stats = {}
        self._first_conv = None
        self.visit(expr)
        self._total_flops = reduce(lambda x, y: x + y, (map(lambda x : x[0], self._stats.values())))
        self._total_memory = reduce(lambda x, y: x + y, (map(lambda x : x[1], self._stats.values())))
        self._l2_norm = reduce(lambda x, y: x + y, (map(lambda x : x[2], self._stats.values())))

    def parse_shape(self, data_shape, kernel_shape, out_shape):
        oc, ic, kh, kw = kernel_shape
        n, test1, ih, iw = data_shape
        _, test2, oh, ow = out_shape

        assert test1 == ic
        assert test2 == oc

        wkl = dict()
        wkl["oc"] = oc
        wkl["ic"] = ic
        wkl["kh"] = kh
        wkl["kw"] = kw
        wkl["n"] = n
        wkl["ih"] = ih
        wkl["iw"] = iw
        wkl["oh"] = oh
        wkl["ow"] = ow
        return wkl


    def visit_call(self, call):
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

        if call.op.name == "nn.conv2d":
            data = call.args[0]
            kernel = call.args[1]

            # Get Shape
            data_shape = get_const_tuple(data.checked_type.shape)
            kernel_shape = get_const_tuple(kernel.checked_type.shape)
            out_shape = get_const_tuple(call.checked_type.shape)
            wkl = self.parse_shape(data_shape, kernel_shape, out_shape)

            # Get params name
            assert isinstance(kernel, tvm.relay.expr.Var)
            param_name = kernel.name_hint
            assert param_name in self._params

            # Checks for assumptions
            assert call.attrs.data_layout == "NCHW"
            assert call.attrs.kernel_layout == "OIHW"
            assert call.attrs.groups == 1
            # FIXME - Add padding, stride, dilations -- all default for now

            compression_method = self._method

            ## If we have 1x1 kernel, we can rely on simple SVD
            ## Drastically improves Tucker
            if wkl["kh"] == 1 and wkl["kw"] == 1 and compression_method != "no_decomp":
                compression_method = "spatial_svd"


            if wkl["ic"] == 3:
                self._first_conv = param_name
                self._first_conv_orig_stats = dict()

                oc, ic, kh, kw = kernel_shape
                n, test1, ih, iw = data_shape
                _, test2, oh, ow = out_shape
                flops = 2 * oc * ic * kh * kw * oh * ow 
                memory = reduce(operator.mul, self._params[param_name].asnumpy().shape, 1)
                l2_norm = np.linalg.norm(self._params[param_name].asnumpy())
                self._first_conv_orig_stats[param_name] = (flops, memory, l2_norm)


            del self._optimized_params[param_name]
            if compression_method == "weight_svd":
                obj = WeightSVD()
            elif compression_method == "spatial_svd":
                obj = SpatialSVD()
            elif compression_method == "tucker_decomp":
                obj = TuckerDecomposition()
            elif compression_method == "tensor_train_decomp":
                obj = TensorTrainDecomposition()
            elif compression_method == "cp_decomp":
                obj = CPDecomposition()
            elif compression_method == "no_decomp":
                obj = NoDecomposition()
            else:
                raise NotImplementedError(compression_method)


            time1 = time.time()
            approx_weight = obj.simulate(self._params[param_name].asnumpy(), wkl,
                                         self._compression_ratio, self._ctx)
            self._optimized_params[param_name] = approx_weight
            self._stats[param_name] = (obj.flops, obj.memory, obj.l2_norm)
            time2 = time.time()
            print("Simulated for", self._compression_ratio, param_name, self._stats[param_name], time2 - time1)
