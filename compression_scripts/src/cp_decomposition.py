from src.base import TensorDecomposition
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, randomised_parafac
import math
import torch
tl.set_backend('pytorch')
device = 'cuda:0'
class CPDecomposition(TensorDecomposition):
    """ Matricize the weight tensor. Perform SVD. """
    def preprocess_tensor(self, tensor):
        # Transform OIHW to IHWO 
        return np.transpose(tensor, axes=(1, 2, 3, 0))

    def postprocess_tensor(self, new_tensor, shape):
        ## Go back to 4D shape
        new_tensor = new_tensor.cpu().detach().numpy()
        return np.transpose(new_tensor, axes=(3, 0, 1, 2))

    def decompose(self, tensor):
        tl_tensor = tl.tensor(tl.tensor(tensor), device=device)
        # Without init random, there is a seg fault, likely memory explosion
        # https://github.com/tensorly/tensorly/issues/68
        factors = parafac(tl_tensor, rank=self.low_rank, init='random')
        return factors

    def reconstruct(self, factors, low_rank):
        new_tensor = tl.kruskal_to_tensor(factors)
        return new_tensor
 
    def simulate(self, tensor, wkl, compression_ratio):
        new_tensor = self.preprocess_tensor(tensor)
        # First get the low rank
        low_rank = self._compute_low_rank(wkl, compression_ratio)

        # Save the rank as APIs are not preserved
        self.low_rank = low_rank

        factors = self.decompose(new_tensor)
        approx_tensor = self.reconstruct(factors, low_rank)
        approx_tensor = self.postprocess_tensor(approx_tensor, tensor.shape)

        self.predict_flops(wkl, low_rank)
        self.predict_memory(wkl, low_rank)
        self.l2_norm = np.linalg.norm(approx_tensor)
        return approx_tensor


    def predict_memory(self, wkl, low_rank):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]
        self.memory = (oc + ic + kh + kw) * low_rank
        
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
        self.flops = 2 * low_rank * oh * ow * (ic + kh + kw + oc) 
 
    def _compute_low_rank(self, wkl, compression_ratio):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]

        original = oc * ic * kh * kw
        new_wo_rank = oc + ic + kh + kw
        low_rank = math.ceil((original * 1.0)/(compression_ratio * new_wo_rank))
        return low_rank
