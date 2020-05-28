from src.base import TensorDecomposition
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, randomised_parafac, partial_tucker
import math
import torch
tl.set_backend('pytorch')
device = 'cuda:0'
device = 'cpu'

class TuckerDecomposition(TensorDecomposition):
    """ Matricize the weight tensor. Perform SVD. """
    def preprocess_tensor(self, tensor):
        # Transform OIHW to HWIO
        return np.transpose(tensor, axes=(2, 3, 1, 0))

    def postprocess_tensor(self, new_tensor, shape):
        ## Go from HWIO to OIHW
        new_tensor = new_tensor.cpu().detach().numpy()
        return np.transpose(new_tensor, axes=(3, 2, 0, 1))

    def decompose(self, tensor):
        tl_tensor = tl.tensor(tl.tensor(tensor), device=device)
        # Without init random, there is a seg fault, likely memory explosion
        # https://github.com/tensorly/tensorly/issues/68
        core, factors = partial_tucker(tl_tensor,
                                       modes=[2, 3],
                                       ranks=[self.low_rank1, self.low_rank2],
                                       init='svd')
        return (core, factors)

    def reconstruct(self, tuckered, low_rank):
        core, factors = tuckered
        # Chain the n-mode product to reconstruct the tensor.
        # There is no straightforward way to come back.
        new_tensor = tl.tenalg.mode_dot(core, factors[0], mode=2)
        new_tensor = tl.tenalg.mode_dot(new_tensor, factors[1], mode=3)
        return new_tensor

    def simulate(self, tensor, wkl, compression_ratio):
        new_tensor = self.preprocess_tensor(tensor)
        # First get the low rank
        self._compute_low_rank(wkl, compression_ratio)

        tuckered = self.decompose(new_tensor)
        approx_tensor = self.reconstruct(tuckered, None)
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
        self.memory = kh * kw * self.low_rank1 * self.low_rank2
        self.memory += oc * self.low_rank2
        self.memory += ic * self.low_rank1

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
        self.flops  = 2 * oh * ow * (kh * kw * self.low_rank1 * self.low_rank2)
        self.flops += 2 * oh * ow * (oc * self.low_rank2)
        self.flops += 2 * oh * ow * (ic * self.low_rank1)

    def _compute_low_rank(self, wkl, compression_ratio):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]


        # FIXME - We can also use VBMF to calculate ranks
        ## https://github.com/CasvandenBogaard/VBMF/blob/master/VBMF.py
        ## https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
        ## Skipping for now as Tucker seems to be doing the worst
        ## We will have rank r1 and tr2
        ## Assume r1/ic = r2/oc then r1 = ic/oc * r2
        coeff = ic * 1.0/oc

        ## this leads to a quadratic equation for r2
        ## Total memory footprint = ic * r1 + oc * r2 + kh * kw * r1 * r2
        ## RHS = (ic * coeff + oc) * r2 + (kh * kw * coeff) * r2^2
        ## LHS = Original/compression_ratio
        original = oc * ic * kh * kw

        ## Forming ax^2 + bx + c = 0
        a = float((kh * kw * coeff))
        b = float((ic * coeff + oc))
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

        r2 = math.ceil(x)
        r1 = math.ceil(coeff * x)
        self.low_rank1 = r1
        self.low_rank2 = r2
