from src.base import TensorDecomposition
import numpy as np
from scipy import linalg
import math

class NoDecomposition(TensorDecomposition):
    """ Matricize the weight tensor. Perform SVD. """
    def preprocess_tensor(self, tensor):
        # Reshape to o * ihw
        out_channels = tensor.shape[0]
        matrix = np.reshape(tensor, newshape=(out_channels, -1))
        return matrix

    def postprocess_tensor(self, matrix, shape):
        ## Go back to 4D shape
        new_tensor = np.reshape(matrix, newshape=shape)
        return new_tensor

    def decompose(self, matrix):
        U, s, Vh = linalg.svd(matrix, full_matrices=False)
        return U, s, Vh

    def reconstruct(self, decomposed, low_rank=None):
        U, s, Vh = decomposed
        m = U.shape[0]
        n = Vh.shape[1]

        if low_rank is None:
            rank = U.shape[1]
        else:
            max_rank = U.shape[1]
            rank = math.ceil(low_rank)

        matrix = np.zeros((m, n))

        for r in range(rank):
            U_col = U[:, r]
            Vh_row = Vh[r, :]
            outer_product = np.outer(U_col, Vh_row)
            matrix = matrix + s[r] * outer_product

        matrix = matrix.astype("float32")
        return matrix
 
    def simulate(self, tensor, wkl, compression_ratio, ctx):
        assert compression_ratio is None
        matrix = self.preprocess_tensor(tensor)
        U, s, Vh = self.decompose(matrix)
        matrix = self.reconstruct((U, s, Vh))
        same_tensor = self.postprocess_tensor(matrix, tensor.shape)

        self.predict_flops(wkl, None)
        self.predict_memory(wkl, None)
        self.l2_norm = np.linalg.norm(same_tensor)
        return same_tensor


    def predict_memory(self, wkl, low_rank):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]
        self.memory = (oc * ic * kh * kw)
        
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

        # Flops are basically two convolutions in sequence
        # (n, ic, ih, iw) * (low_rank, ic, kh, kw) --> (n, low_rank, oh, ow)
        self.flops = 2 * oc * ic * kh * kw * oh * ow 
