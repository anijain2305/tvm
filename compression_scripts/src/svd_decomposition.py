from src.base import TensorDecomposition
import numpy as np
from scipy import linalg
import math

class WeightSVD(TensorDecomposition):
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

    def reconstruct(self, decomposed, low_rank):
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
 
    def simulate(self, tensor, wkl, compression_ratio):
        matrix = self.preprocess_tensor(tensor)
        U, s, Vh = self.decompose(matrix)
        low_rank = self._compute_low_rank(wkl, compression_ratio)
        approx_matrix = self.reconstruct((U, s, Vh), low_rank)
        approx_tensor = self.postprocess_tensor(approx_matrix, tensor.shape)

        self.predict_flops(wkl, low_rank)
        self.predict_memory(wkl, low_rank)
        self.l2_norm = np.linalg.norm(approx_tensor)
        return approx_tensor


    def predict_memory(self, wkl, low_rank):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]
        self.memory = (oc + ic * kh * kw) * low_rank
        
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
        self.flops = 2 * low_rank * ic * kh * kw * oh * ow 
        # (n, low_rank, oh', ow') * (oc, low_rank, 1, 1) --> (n, oc, oh, ow)
        self.flops += 2 * oc * low_rank * 1 * 1 * oh * ow
 
    def _compute_low_rank(self, wkl, compression_ratio):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]

        original = oc * ic * kh * kw
        new_wo_rank = oc + ic * kh * kw
        low_rank = (original * 1.0)/(compression_ratio * new_wo_rank)
        return low_rank


class SpatialSVD(TensorDecomposition):
    """ Matricize the weight tensor. Perform SVD. """
    def preprocess_tensor(self, tensor):
        # Go from OIHW to IHWO
        oc, ic, kh, kw = tensor.shape
        tensor = np.transpose(tensor, axes=(1, 2, 3, 0))

        # Reshape to IH * WO
        matrix = np.reshape(tensor, newshape=(ic * kh, kw * oc))
        return matrix

    def postprocess_tensor(self, matrix, shape):
        ## Go back to 4D shape
        oc, ic, kh, kw = shape

        # Go back to I * H * W * O from IH * WO
        new_tensor = np.reshape(matrix, newshape=(ic, kh, kw, oc))

        # Go back to OIHW
        new_tensor = np.transpose(new_tensor, axes=(3, 0, 1, 2))
        return new_tensor

    def decompose(self, matrix):
        U, s, Vh = linalg.svd(matrix, full_matrices=False)
        return U, s, Vh

    def reconstruct(self, decomposed, low_rank):
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
 
    def simulate(self, tensor, wkl, compression_ratio):
        matrix = self.preprocess_tensor(tensor)
        U, s, Vh = self.decompose(matrix)
        low_rank = self._compute_low_rank(wkl, compression_ratio)
        approx_matrix = self.reconstruct((U, s, Vh), low_rank)
        approx_tensor = self.postprocess_tensor(approx_matrix, tensor.shape)

        self.predict_flops(wkl, low_rank)
        self.predict_memory(wkl, low_rank)
        self.l2_norm = np.linalg.norm(approx_tensor)
        return approx_tensor


    def predict_memory(self, wkl, low_rank):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]
        # IH * WO
        self.memory = (ic * kh + kw * oc) * low_rank
        
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
        # IH * r --> (n, ic, ih, iw) * (rank, ic, kh, 1) --> (n, rank, oh, iw)
        self.flops = 2 * low_rank * ic * kh * 1 * oh * iw 
        # r * WO --> (n, rank, oh, iw) * (oc, rank, 1, kw) --> (n, oc, oh, ow)
        self.flops += 2 * oc * low_rank * 1 * kw * oh * ow
 
    def _compute_low_rank(self, wkl, compression_ratio):
        oc = wkl["oc"]
        ic = wkl["ic"]
        kh = wkl["kh"]
        kw = wkl["kw"]

        original = oc * ic * kh * kw
        # IH * WO
        new_wo_rank = ic * kh + kw * oc
        low_rank = (original * 1.0)/(compression_ratio * new_wo_rank)
        return low_rank
