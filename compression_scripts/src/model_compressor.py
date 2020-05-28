import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from topi.util import get_const_tuple

from functools import reduce
from src.svd_decomposition import WeightSVD, SpatialSVD 
from src.no_decomposition import NoDecomposition 
from src.cp_decomposition import CPDecomposition 
from src.tucker_decomposition import TuckerDecomposition
from src.tensor_train_decomposition import TensorTrainDecomposition
import time

class ModelCompressor(ExprVisitor):
    def __init__(self):
        super().__init__()
        self._stats = {}

    def compress(self, params, expr, compression_ratio, method):
        self._params = params
        self._optimized_params = dict(params)
        self._compression_ratio = compression_ratio
        self._method = method
        self._stats = {}
        self.visit(expr)
        self._total_memory = reduce(lambda x, y: x + y, (map(lambda x : x[0], self._stats.values())))
        self._total_flops = reduce(lambda x, y: x + y, (map(lambda x : x[1], self._stats.values())))
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
            del self._optimized_params[param_name]

            # Checks for assumptions
            assert call.attrs.data_layout == "NCHW"
            assert call.attrs.kernel_layout == "OIHW"
            assert call.attrs.groups == 1
            # FIXME - Add padding, stride, dilations -- all default for now

            skip = False
            if self._method == "tucker_decomp":
                if data_shape[1] == 3:
                    skip = True
                elif wkl['kh'] == 1:
                    skip = True

            if self._method == "weight_svd":
                obj = WeightSVD()
            elif self._method == "spatial_svd":
                obj = SpatialSVD()
            elif self._method == "tucker_decomp":
                obj = TuckerDecomposition()
            elif self._method == "tensor_train_decomp":
                obj = TensorTrainDecomposition()
            elif self._method == "cp_decomp":
                obj = CPDecomposition()
            elif self._method == "no_decomp":
                obj = NoDecomposition()
            else:
                raise NotImplementedError(self._method)


            time1 = time.time()
            approx_weight = obj.simulate(self._params[param_name].asnumpy(), wkl, self._compression_ratio)
            self._optimized_params[param_name] = approx_weight
            self._stats[param_name] = (obj.flops, obj.memory, obj.l2_norm)
            time2 = time.time()
            print("Simulated for ", param_name, self._stats[param_name], time2 - time1)


