"""Basic SAFE matrices, port of spectrum/routines/BasicMatrices/*.m."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import convolve

from .structures import AttrDict


def convolution_cache_dir():
    """Directory for the reusable convolution integral caches.

    The convolution integrals depend only on the element orders, so they are
    cached between frequencies and runs. The cache lives in the system temp
    directory (not in the output folder, which keeps only the results).
    """
    return Path(tempfile.gettempdir()) / "aniso_safe_cache"


def L1L2_int_matrix(N_degree):
    """Compute edge integral coefficients a!b!/(a+b+1)!, port of L1L2_int_matrix.m."""
    n = int(N_degree)
    out = np.zeros((n, n), dtype=float)
    for aa in range(1, n + 1):
        for bb in range(aa, n + 1):
            a = aa - 1
            b = bb - 1
            value = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 1)
            out[aa - 1, bb - 1] = value
            out[bb - 1, aa - 1] = value
    return out


def L1L2L3_int_matrix(N_degree):
    """Compute triangle integral coefficients 2*a!b!c!/(a+b+c+2)!, port of L1L2L3_int_matrix.m."""
    n = int(N_degree)
    out = np.zeros((n, n, n), dtype=float)
    for aa in range(1, n + 1):
        for bb in range(aa, n + 1):
            for cc in range(bb, n + 1):
                a = aa - 1
                b = bb - 1
                c = cc - 1
                value = 2.0 * math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 2)
                for perm in ((aa, bb, cc), (aa, cc, bb), (bb, aa, cc), (bb, cc, aa), (cc, aa, bb), (cc, bb, aa)):
                    out[perm[0] - 1, perm[1] - 1, perm[2] - 1] = value
    return out


def NL_matrix():
    """Expand cubic triangle shape functions into L1/L2/L3 monomials, port of NL_matrix.m."""
    nl = np.zeros((10, 4, 4, 4), dtype=float)
    node_l_coord = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [0.0, 2.0 / 3.0, 1.0 / 3.0],
            [0.0, 1.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 0.0, 1.0 / 3.0],
            [1.0 / 3.0, 0.0, 2.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )

    nl[0, 1, 0, 0] = 1.0
    nl[0, 2, 0, 0] = -9.0 / 2.0
    nl[0, 3, 0, 0] = 9.0 / 2.0
    nl[1, 0, 1, 0] = 1.0
    nl[1, 0, 2, 0] = -9.0 / 2.0
    nl[1, 0, 3, 0] = 9.0 / 2.0
    nl[2, 0, 0, 1] = 1.0
    nl[2, 0, 0, 2] = -9.0 / 2.0
    nl[2, 0, 0, 3] = 9.0 / 2.0

    nl[3, 1, 1, 0] = -9.0 / 2.0
    nl[3, 2, 1, 0] = 27.0 / 2.0
    nl[4, 1, 1, 0] = -9.0 / 2.0
    nl[4, 1, 2, 0] = 27.0 / 2.0
    nl[5, 0, 1, 1] = -9.0 / 2.0
    nl[5, 0, 2, 1] = 27.0 / 2.0
    nl[6, 0, 1, 1] = -9.0 / 2.0
    nl[6, 0, 1, 2] = 27.0 / 2.0
    nl[7, 1, 0, 1] = -9.0 / 2.0
    nl[7, 1, 0, 2] = 27.0 / 2.0
    nl[8, 1, 0, 1] = -9.0 / 2.0
    nl[8, 2, 0, 1] = 27.0 / 2.0
    nl[9, 1, 1, 1] = 27.0
    return nl, node_l_coord


def dNL_matrices(BasicMatrices):
    """Compute derivative expansion matrices, port of dNL_matrices.m."""
    dnl = np.zeros((3, 10, 4, 4, 4), dtype=float)
    dnl_val = np.zeros((3, 10, 10), dtype=float)
    nl = BasicMatrices.NLMatrix
    coord = BasicMatrices.NodeLCoord
    for nN in range(10):
        for ii in range(1, 4):
            for jj in range(1, 4):
                for kk in range(1, 4):
                    dnl[0, nN, ii - 1, jj - 1, kk - 1] = ii * nl[nN, ii, jj - 1, kk - 1]
                    dnl[1, nN, ii - 1, jj - 1, kk - 1] = jj * nl[nN, ii - 1, jj, kk - 1]
                    dnl[2, nN, ii - 1, jj - 1, kk - 1] = kk * nl[nN, ii - 1, jj - 1, kk]
                    for nN2 in range(10):
                        factor = (
                            coord[nN2, 0] ** (ii - 1)
                            * coord[nN2, 1] ** (jj - 1)
                            * coord[nN2, 2] ** (kk - 1)
                        )
                        dnl_val[:, nN, nN2] += dnl[:, nN, ii - 1, jj - 1, kk - 1] * factor
    return dnl, dnl_val


def NLEdge_matrix():
    """Expand cubic edge shape functions into L1/L2 monomials, port of NLEdge_matrix.m."""
    out = np.zeros((4, 4, 4), dtype=float)
    out[0, 1, 0] = 1.0
    out[0, 2, 0] = -9.0 / 2.0
    out[0, 3, 0] = 9.0 / 2.0
    out[1, 0, 1] = 1.0
    out[1, 0, 2] = -9.0 / 2.0
    out[1, 0, 3] = 9.0 / 2.0
    out[2, 1, 1] = -9.0 / 2.0
    out[2, 2, 1] = 27.0 / 2.0
    out[3, 1, 1] = -9.0 / 2.0
    out[3, 1, 2] = 27.0 / 2.0
    return out


def _conv3(a, b):
    """Full 3D convolution equivalent to MATLAB convn for 3D arrays."""
    return convolve(a, b, mode="full", method="auto")


def ConvolveMatrices(CompStruct, BasicMatrices, cache_dir=None):
    """Compute double convolution integrals, port of ConvolveMatrices.m."""
    n_nodes = int(CompStruct.Advanced.N_nodes)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"ConvMatrices_N{n_nodes}.npz"
        if cache_file.exists():
            try:
                with np.load(cache_file) as data:
                    for name in ("NNNConvMatrixInt", "dNNNConvMatrixInt", "NNdNConvMatrixInt", "dNNdNConvMatrixInt"):
                        BasicMatrices[name] = data[name]
                _expand_convolution_large(CompStruct, BasicMatrices)
                return BasicMatrices
            except Exception:
                # the cache is best-effort only; recompute on any problem
                pass

    nnn = np.zeros((n_nodes, n_nodes, n_nodes, 10, 10, 10), dtype=float)
    dnnn = np.zeros((3, n_nodes, n_nodes, n_nodes, 10, 10, 10), dtype=float)
    nndn = np.zeros((3, n_nodes, n_nodes, n_nodes, 10, 10, 10), dtype=float)
    dnndn = np.zeros((3, 3, n_nodes, n_nodes, n_nodes, 10, 10, 10), dtype=float)
    lint = BasicMatrices.LIntMatrix9[:10, :10, :10]

    for bb in range(n_nodes):
        for cc in range(n_nodes):
            nb = BasicMatrices.NLMatrix[bb]
            nc = BasicMatrices.NLMatrix[cc]
            conv_nb_nc = _conv3(nb, nc)
            for aa in range(n_nodes):
                na = BasicMatrices.NLMatrix[aa]
                nnn[aa, bb, cc] = _conv3(na, conv_nb_nc)
                for ii in range(3):
                    dna_i = BasicMatrices.dNLMatrix[ii, aa]
                    dnnn[ii, aa, bb, cc] = _conv3(dna_i, conv_nb_nc)
            for jj in range(3):
                dnc_j = BasicMatrices.dNLMatrix[jj, cc]
                conv_nb_dnc = _conv3(nb, dnc_j)
                for aa in range(n_nodes):
                    na = BasicMatrices.NLMatrix[aa]
                    nndn[jj, aa, bb, cc] = _conv3(na, conv_nb_dnc)
                    for ii in range(3):
                        dna_i = BasicMatrices.dNLMatrix[ii, aa]
                        dnndn[ii, jj, aa, bb, cc] = _conv3(dna_i, conv_nb_dnc)

    BasicMatrices.NNNConvMatrixInt = np.sum(nnn * lint, axis=(3, 4, 5))
    BasicMatrices.dNNNConvMatrixInt = np.sum(dnnn * lint, axis=(4, 5, 6))
    BasicMatrices.NNdNConvMatrixInt = np.sum(nndn * lint, axis=(4, 5, 6))
    BasicMatrices.dNNdNConvMatrixInt = np.sum(dnndn * lint, axis=(5, 6, 7))

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_file,
                NNNConvMatrixInt=BasicMatrices.NNNConvMatrixInt,
                dNNNConvMatrixInt=BasicMatrices.dNNNConvMatrixInt,
                NNdNConvMatrixInt=BasicMatrices.NNdNConvMatrixInt,
                dNNdNConvMatrixInt=BasicMatrices.dNNdNConvMatrixInt,
            )
        except OSError:
            # caching must never break the computation itself
            pass
    _expand_convolution_large(CompStruct, BasicMatrices)
    return BasicMatrices


def _expand_convolution_large(CompStruct, BasicMatrices):
    """Expand convolution blocks over domain variables, port of the final loop in ConvolveMatrices.m."""
    n_nodes = int(CompStruct.Advanced.N_nodes)
    BasicMatrices.dNNdNConvMatrixIntLarge = []
    BasicMatrices.dNNNConvMatrixIntLarge = []
    BasicMatrices.NNdNConvMatrixIntLarge = []
    BasicMatrices.NNNConvMatrixIntLarge = []
    for ii_d in range(int(CompStruct.Data.N_domain)):
        dvar = int(CompStruct.Data.DVarNum[ii_d])
        ones_var = np.ones((dvar, dvar), dtype=float)
        msize = dvar * n_nodes
        dnndn_large = np.zeros((3, 3, msize, n_nodes, msize), dtype=float)
        dnnn_large = np.zeros((3, msize, n_nodes, msize), dtype=float)
        nndn_large = np.zeros((3, msize, n_nodes, msize), dtype=float)
        nnn_large = np.zeros((msize, n_nodes, msize), dtype=float)
        for ii_k in range(n_nodes):
            nnn_large[:, ii_k, :] = np.kron(BasicMatrices.NNNConvMatrixInt[:, ii_k, :], ones_var)
            for ii_c in range(3):
                nndn_large[ii_c, :, ii_k, :] = np.kron(BasicMatrices.NNdNConvMatrixInt[ii_c, :, ii_k, :], ones_var)
                dnnn_large[ii_c, :, ii_k, :] = np.kron(BasicMatrices.dNNNConvMatrixInt[ii_c, :, ii_k, :], ones_var)
                for ii_c2 in range(3):
                    dnndn_large[ii_c, ii_c2, :, ii_k, :] = np.kron(
                        BasicMatrices.dNNdNConvMatrixInt[ii_c, ii_c2, :, ii_k, :], ones_var
                    )
        BasicMatrices.dNNdNConvMatrixIntLarge.append(dnndn_large)
        BasicMatrices.dNNNConvMatrixIntLarge.append(dnnn_large)
        BasicMatrices.NNdNConvMatrixIntLarge.append(nndn_large)
        BasicMatrices.NNNConvMatrixIntLarge.append(nnn_large)


def ConvolveEdgeMatrices(CompStruct, BasicMatrices, cache_dir=None):
    """Compute edge convolution integrals, port of ConvolveEdgeMatrices.m."""
    n_edge = int(CompStruct.Advanced.NEdge_nodes)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"ConvEdgeMatrices_N{n_edge}.npz"
        if cache_file.exists():
            try:
                BasicMatrices.NENENEConvMatrixInt = np.load(cache_file)["NENENEConvMatrixInt"]
                return BasicMatrices
            except Exception:
                # the cache is best-effort only; recompute on any problem
                pass

    nenene = np.zeros((n_edge, n_edge, n_edge, 10, 10), dtype=float)
    lint_edge = BasicMatrices.LEdgeIntMatrix9[:10, :10]
    for bb in range(n_edge):
        for cc in range(n_edge):
            neb = BasicMatrices.NLEdgeMatrix[bb]
            nec = BasicMatrices.NLEdgeMatrix[cc]
            conv_neb_nec = convolve(neb, nec, mode="full", method="auto")
            for aa in range(n_edge):
                nea = BasicMatrices.NLEdgeMatrix[aa]
                nenene[aa, bb, cc] = convolve(nea, conv_neb_nec, mode="full", method="auto")
    BasicMatrices.NENENEConvMatrixInt = np.sum(nenene * lint_edge, axis=(3, 4))
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, NENENEConvMatrixInt=BasicMatrices.NENENEConvMatrixInt)
        except OSError:
            # caching must never break the computation itself
            pass
    return BasicMatrices


def AssembleBasicMatrices_sp_SAFE(CompStruct, cache_dir=None):
    """Assemble basic matrices, port of AssembleBasicMatrices_sp_SAFE.m."""
    basic = AttrDict()
    basic.Lx = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    basic.Ly = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    basic.Lz = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    basic.Lx_fluid = np.array([[1], [0], [0]], dtype=float)
    basic.Ly_fluid = np.array([[0], [1], [0]], dtype=float)
    basic.Lz_fluid = np.array([[0], [0], [1]], dtype=float)
    basic.E3 = np.eye(3)
    basic.LEdgeIntMatrix9 = L1L2_int_matrix(CompStruct.Advanced.N_nodes)
    basic.LIntMatrix9 = L1L2L3_int_matrix(CompStruct.Advanced.N_nodes)
    basic.NLMatrix, basic.NodeLCoord = NL_matrix()
    basic.dNLMatrix, basic.dNLMatrix_val = dNL_matrices(basic)
    basic = ConvolveMatrices(CompStruct, basic, cache_dir=cache_dir)
    basic.NLEdgeMatrix = NLEdge_matrix()
    basic = ConvolveEdgeMatrices(CompStruct, basic, cache_dir=cache_dir)
    return basic
