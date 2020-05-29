from src.base import TensorDecomposition
import numpy as np
import tensorly as tl
from tensorly.decomposition import matrix_product_state
import math
import torch
tl.set_backend('pytorch')
device = 'cuda:0'
device = 'cpu'

class TensorTrainDecomposition(TensorDecomposition):
    """ Matricize the weight tensor. Perform SVD. """
    def preprocess_tensor(self, tensor):
        # Transform OIHW to IHWO
        return np.transpose(tensor, axes=(1, 2, 3, 0))

    def postprocess_tensor(self, new_tensor, shape):
        ## Go from IHWO to OIHW
        new_tensor = new_tensor.cpu().detach().numpy()
        return np.transpose(new_tensor, axes=(3, 0, 1, 2))

    def decompose(self, tensor):
        tl_tensor = tl.tensor(tl.tensor(tensor), device=device)
        # Without init random, there is a seg fault, likely memory explosion
        # https://github.com/tensorly/tensorly/issues/68
        factors = matrix_product_state(tl_tensor,
                                       rank=[1, self.low_rank1, self.low_rank2, self.low_rank3, 1])
        return factors

    def reconstruct(self, factors, low_rank):
        new_tensor = tl.mps_to_tensor(factors)
        return new_tensor

    def simulate(self, tensor, wkl, compression_ratio):
        new_tensor = self.preprocess_tensor(tensor)
        # First get the low rank
        self._compute_low_rank(wkl, compression_ratio)

        factors = self.decompose(new_tensor)
        approx_tensor = self.reconstruct(factors, None)
        approx_tensor = self.postprocess_tensor(approx_tensor, tensor.shape)

        self.predict_flops(wkl, None)
        self.predict_memory(wkl, None)
        self.l2_norm = np.linalg.norm(approx_tensor)
        return approx_tensor


    def predict_memory(self, wkl, low_rank):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]
        self.memory = ic * self.low_rank1
        self.memory += self.low_rank1 * kh * self.low_rank2
        self.memory += self.low_rank2 * kw * self.low_rank3
        self.memory += self.low_rank3 * oc

    def predict_flops(self, wkl, low_rank):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]
        n = wkl["n"]
        oh = wkl["oh"]
        ow = wkl["ow"]
        ih = wkl["ih"]
        iw = wkl["iw"]

        # Flops are basically four convolutions in sequence
        self.flops  = 2 * oh * ow * (ic * self.low_rank1)
        self.flops += 2 * oh * ow * (self.low_rank1 * self.low_rank2 * kh)
        self.flops += 2 * oh * ow * (self.low_rank2 * self.low_rank3 * kw)
        self.flops += 2 * oh * ow * (self.low_rank3 * oc)

    def _compute_low_rank(self, wkl, compression_ratio):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]


        # FIXME - We can also use VBMF to calculate ranks
        ## https://github.com/CasvandenBogaard/VBMF/blob/master/VBMF.py
        ## https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
        ## We will have rank r1, r2 and r3
        ## Assume r1/ic = r2/kh = r3/oc then r1 = ic/oc * r3, r2 = kh/oc * r3
        max_r1 = ic
        max_r2 = kh * max_r1
        max_r3 = kw * max_r2
        c1 = max_r1/max_r2
        c2 = max_r2/max_r3

        ## this leads to a quadratic equation for r2
        ## Total memory footprint = ic * r1 + kh * r1 * r2 + kw * r2 * r3 + oc * r3
        ## LHS = Original/compression_ratio
        original = oc * ic * kh * kw

        ## Forming ax^2 + bx + c = 0
        a = float((kw * c2 + kh * c1 * c2))
        b = float((ic * c1 + oc))
        c = float(-original/compression_ratio)

        r = b**2 - 4*a*c

        if r > 0:
            num_roots = 2
            x1 = (((-b) + math.sqrt(r))/(2*a))
            x2 = (((-b) - math.sqrt(r))/(2*a))
            x = x1
        elif r == 0:
            num_roots = 1
            x1 = (-b) / 2*a
            x = x1
            print("There is one root: ", x)
        else:
            assert False

        r3 = math.ceil(x)
        r2 = math.ceil(c2 * x)
        r1 = math.ceil(c1 * x)
        self.low_rank1 = r1
        self.low_rank2 = r2
        self.low_rank3 = r3
