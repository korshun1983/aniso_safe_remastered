"""Finite-element matrix assembly, port of spectrum/routines/FEMatrices.

All functions keep the MATLAB names and the computational structure of the
original M-files. Array indices follow the Python zero-based convention;
the positions where the MATLAB code relies on one-based indices are marked
in the comments. MATLAB function handles (CompStruct.Methods.*) are replaced
by the name-based METHODS registry and the call_method dispatcher.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .structures import AttrDict


METHODS = {}


def _register(name):
    """Register a ported function under its MATLAB name."""

    def decorator(func):
        METHODS[name] = func
        return func

    return decorator


def call_method(name, *args):
    """Call a ported method by its MATLAB name (replaces function handles)."""
    func = METHODS.get(name)
    if func is None:
        raise NotImplementedError(f"Method '{name}' is not ported yet")
    return func(*args)


# ============================================================================
# Elastic moduli tensor routines (port of routines/em_tensor_VTI.m,
# routines/rot_matrix.m, routines/rot_c_ij.m)
# ============================================================================


@_register("em_tensor_VTI")
def em_tensor_VTI(c_VTI):
    """Populate elastic moduli tensors from VTI parameters, port of em_tensor_VTI.m."""
    c_VTI = np.asarray(c_VTI, dtype=float).ravel()
    # Voigt notation (c_IJ)
    c_ij = np.zeros((6, 6), dtype=float)
    c_ij[0, 0] = c_VTI[0]
    c_ij[1, 1] = c_VTI[0]
    c_ij[2, 2] = c_VTI[2]
    c_ij[3, 3] = c_VTI[3]
    c_ij[4, 4] = c_VTI[3]
    c_ij[5, 5] = c_VTI[4]
    c_ij[0, 1] = c_ij[0, 0] - 2.0 * c_ij[5, 5]
    c_ij[0, 2] = c_VTI[1]
    c_ij[1, 2] = c_VTI[1]
    c_ij[1, 0] = c_ij[0, 1]
    c_ij[2, 0] = c_ij[0, 2]
    c_ij[2, 1] = c_ij[1, 2]

    # Full formulation notation (c_ijkl)
    c_ijkl = np.zeros((3, 3, 3, 3), dtype=float)
    c_ijkl[0, 0, 0, 0] = c_ij[0, 0]
    c_ijkl[1, 1, 1, 1] = c_ij[1, 1]
    c_ijkl[2, 2, 2, 2] = c_ij[2, 2]
    c_ijkl[0, 0, 1, 1] = c_ij[0, 1]
    c_ijkl[1, 1, 0, 0] = c_ij[1, 0]
    c_ijkl[0, 0, 2, 2] = c_ij[0, 2]
    c_ijkl[2, 2, 0, 0] = c_ij[2, 0]
    c_ijkl[1, 1, 2, 2] = c_ij[1, 2]
    c_ijkl[2, 2, 1, 1] = c_ij[2, 1]
    c_ijkl[2, 1, 2, 1] = c_ij[3, 3]
    c_ijkl[2, 1, 1, 2] = c_ij[3, 3]
    c_ijkl[1, 2, 2, 1] = c_ij[3, 3]
    c_ijkl[1, 2, 1, 2] = c_ij[3, 3]
    c_ijkl[2, 0, 2, 0] = c_ij[4, 4]
    c_ijkl[2, 0, 0, 2] = c_ij[4, 4]
    c_ijkl[0, 2, 2, 0] = c_ij[4, 4]
    c_ijkl[0, 2, 0, 2] = c_ij[4, 4]
    c_ijkl[0, 1, 0, 1] = c_ij[5, 5]
    c_ijkl[0, 1, 1, 0] = c_ij[5, 5]
    c_ijkl[1, 0, 0, 1] = c_ij[5, 5]
    c_ijkl[1, 0, 1, 0] = c_ij[5, 5]
    return c_ij, c_ijkl


@_register("rot_matrix")
def rot_matrix(theta=0.0, phi=0.0):
    """Compute the rotation matrix from Euler angles, port of rot_matrix.m."""
    theta = float(theta)
    phi = float(phi)
    a_rot = np.array(
        [
            [np.cos(phi), np.sin(phi), 0.0],
            [-np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi), np.sin(theta)],
            [np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), np.cos(theta)],
        ],
        dtype=float,
    )
    return a_rot


@_register("rot_c_ij")
def rot_c_ij(c_ij, rot_m):
    """Rotate the elastic moduli tensor, port of rot_c_ij.m."""
    a = np.asarray(rot_m, dtype=float)
    c_ij = np.asarray(c_ij, dtype=float)

    mincl = np.array(
        [
            [a[0, 0] ** 2, a[0, 1] ** 2, a[0, 2] ** 2,
             2 * a[0, 1] * a[0, 2], 2 * a[0, 0] * a[0, 2], 2 * a[0, 0] * a[0, 1]],
            [a[1, 0] ** 2, a[1, 1] ** 2, a[1, 2] ** 2,
             2 * a[1, 1] * a[1, 2], 2 * a[1, 0] * a[1, 2], 2 * a[1, 0] * a[1, 1]],
            [a[2, 0] ** 2, a[2, 1] ** 2, a[2, 2] ** 2,
             2 * a[2, 1] * a[2, 2], 2 * a[2, 0] * a[2, 2], 2 * a[2, 0] * a[2, 1]],
            [a[1, 0] * a[2, 0], a[1, 1] * a[2, 1], a[1, 2] * a[2, 2],
             a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1],
             a[1, 0] * a[2, 2] + a[1, 2] * a[2, 0],
             a[1, 0] * a[2, 1] + a[1, 1] * a[2, 0]],
            [a[0, 0] * a[2, 0], a[0, 1] * a[2, 1], a[0, 2] * a[2, 2],
             a[0, 1] * a[2, 2] + a[0, 2] * a[2, 1],
             a[0, 0] * a[2, 2] + a[0, 2] * a[2, 0],
             a[0, 0] * a[2, 1] + a[0, 1] * a[2, 0]],
            [a[0, 0] * a[1, 0], a[0, 1] * a[1, 1], a[0, 2] * a[1, 2],
             a[0, 1] * a[1, 2] + a[0, 2] * a[1, 1],
             a[0, 0] * a[1, 2] + a[0, 2] * a[1, 0],
             a[0, 0] * a[1, 1] + a[0, 1] * a[1, 0]],
        ],
        dtype=float,
    )

    nincl = np.array(
        [
            [a[0, 0] ** 2, a[0, 1] ** 2, a[0, 2] ** 2,
             a[0, 1] * a[0, 2], a[0, 0] * a[0, 2], a[0, 0] * a[0, 1]],
            [a[1, 0] ** 2, a[1, 1] ** 2, a[1, 2] ** 2,
             a[1, 1] * a[1, 2], a[1, 0] * a[1, 2], a[1, 0] * a[1, 1]],
            [a[2, 0] ** 2, a[2, 1] ** 2, a[2, 2] ** 2,
             a[2, 1] * a[2, 2], a[2, 0] * a[2, 2], a[2, 0] * a[2, 1]],
            [2 * a[1, 0] * a[2, 0], 2 * a[1, 1] * a[2, 1], 2 * a[1, 2] * a[2, 2],
             a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1],
             a[1, 0] * a[2, 2] + a[1, 2] * a[2, 0],
             a[1, 0] * a[2, 1] + a[1, 1] * a[2, 0]],
            [2 * a[0, 0] * a[2, 0], 2 * a[0, 1] * a[2, 1], 2 * a[0, 2] * a[2, 2],
             a[0, 1] * a[2, 2] + a[0, 2] * a[2, 1],
             a[0, 0] * a[2, 2] + a[0, 2] * a[2, 0],
             a[0, 0] * a[2, 1] + a[0, 1] * a[2, 0]],
            [2 * a[0, 0] * a[1, 0], 2 * a[0, 1] * a[1, 1], 2 * a[0, 2] * a[1, 2],
             a[0, 1] * a[1, 2] + a[0, 2] * a[1, 1],
             a[0, 0] * a[1, 2] + a[0, 2] * a[1, 0],
             a[0, 0] * a[1, 1] + a[0, 1] * a[1, 0]],
        ],
        dtype=float,
    )

    c_ij_rot = (mincl @ c_ij) @ mincl.T
    c_ij_rot_back = (nincl.T @ c_ij) @ nincl
    return c_ij_rot, c_ij_rot_back


@_register("V_phase_VTI_exact_RPH")
def V_phase_VTI_exact_RPH(rho, c_main, theta):
    """Phase velocities of the three bulk plane waves in a VTI medium.

    Port of routines/V_phase_VTI_exact_RPH.m: exact solution of the
    Kelvin-Christoffel equation for a VTI medium (Mavko et al., Rock
    Physics Handbook) for the propagation direction inclined by theta
    to the VTI symmetry axis.

    rho - density in kg/m^3; c_main - [c11 c13 c33 c44 c66] in GPa;
    theta - inclination to the VTI axis in radians.
    Returns (V_qP, V_qSV, V_SH) in m/s.
    """
    c11, c13, c33, c44, c66 = np.asarray(c_main, dtype=float)
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2

    m_th = ((c11 - c44) * sin2 - (c33 - c44) * cos2) ** 2 + (c13 + c44) ** 2 * np.sin(2 * theta) ** 2

    a_v_qp = 0.5 * (c11 * sin2 + c33 * cos2 + c44 + np.sqrt(m_th))
    a_v_qsv = 0.5 * (c11 * sin2 + c33 * cos2 + c44 - np.sqrt(m_th))
    a_v_sh = c66 * sin2 + c44 * cos2

    # 1e+9 factor just translates c_main from GPa into Pa as required by the formula
    v_qp = np.sqrt(1.0e9 * a_v_qp / rho)
    v_qsv = np.sqrt(1.0e9 * a_v_qsv / rho)
    v_sh = np.sqrt(1.0e9 * a_v_sh / rho)
    return v_qp, v_qsv, v_sh


# ============================================================================
# Physical properties preparation (port of PreparePhysProp_* and getPhysProps_*)
# ============================================================================


@_register("PreparePhysProp_fluid_sp_SAFE")
def PreparePhysProp_fluid_sp_SAFE(CompStruct, ii_d):
    """Extract ideal fluid layer parameters, port of PreparePhysProp_fluid_sp_SAFE.m."""
    phys_prop = AttrDict()
    phys_prop.rho = CompStruct.Model.DomainParam[ii_d][0]
    phys_prop["lambda"] = CompStruct.Model.DomainParam[ii_d][1]
    return phys_prop


@_register("PreparePhysProp_HTTI_sp_SAFE")
def PreparePhysProp_HTTI_sp_SAFE(CompStruct, ii_d):
    """Extract VTI layer parameters and rotate the tensor, port of PreparePhysProp_HTTI_sp_SAFE.m."""
    param = np.asarray(CompStruct.Model.DomainParam[ii_d], dtype=float).ravel()
    phys_prop = AttrDict()
    phys_prop.rho = param[0]
    phys_prop.c_VTI = param[1:6]
    phys_prop.theta = param[6]
    phys_prop.phi = param[7] if param.size == 8 else 0.0

    # Computing rotated elastic moduli tensor
    c_ij, _c_ijkl = em_tensor_VTI(phys_prop.c_VTI)
    rot_m = rot_matrix(phys_prop.theta, phys_prop.phi)
    # rotating back to get elastic moduli tensor in the borehole system
    _c_ij_rot, c_ij_rot_back = rot_c_ij(c_ij, rot_m)
    phys_prop.c_ij = c_ij_rot_back
    return phys_prop


@_register("getPhysProps_fluid")
def getPhysProps_fluid(CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el):
    """Interpolate fluid properties inside the element, port of getPhysProps_fluid.m."""
    n_nodes = int(CompStruct.Advanced.N_nodes)
    # retrieve elastic moduli tensor and the density for the domain
    lambda_value = float(FEMatrices.PhysProp[ii_d]["lambda"]) * 1.0e9
    rho_value = float(FEMatrices.PhysProp[ii_d].rho) * 1.0e3
    rho2_lambda_value = rho_value**2 / lambda_value

    el_phys_props = AttrDict()
    el_phys_props.LambdaVec = np.full(n_nodes, lambda_value, dtype=float)
    el_phys_props.RhoVec = np.full(n_nodes, rho_value, dtype=float)
    el_phys_props.Rho2LambdaVec = np.full(n_nodes, rho2_lambda_value, dtype=float)
    return el_phys_props


@_register("getPhysProps_HTTI")
def getPhysProps_HTTI(CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el):
    """Interpolate HTTI properties inside the element, port of getPhysProps_HTTI.m."""
    n_nodes = int(CompStruct.Advanced.N_nodes)
    # retrieve elastic moduli tensor and the density for the domain
    cij_6x6 = np.asarray(FEMatrices.PhysProp[ii_d].c_ij, dtype=float) * 1.0e9
    rho_value = float(FEMatrices.PhysProp[ii_d].rho) * 1.0e3

    el_phys_props = AttrDict()
    el_phys_props.CijMatrix = np.broadcast_to(cij_6x6, (n_nodes, 6, 6)).copy()
    el_phys_props.RhoVec = np.full(n_nodes, rho_value, dtype=float)
    return el_phys_props


# ============================================================================
# Shape function derivative expansions (port of dxNL_matrix.m / dyNL_matrix.m)
# ============================================================================


def _dNL_matrix_weighted(weights, NLMatrix):
    """Shared body of dxNL_matrix.m / dyNL_matrix.m with the given L-weights."""
    # d/dx L_i = 1/(2*delta) * b_i; the factor 1/delta is omitted for the
    # simplification of computations (it is restored in KM_matrix_*)
    d_l = 0.5 * np.asarray(weights, dtype=float)
    ii_w = np.arange(1, 4, dtype=float)
    # dxNLMatrix(nN,ii,jj,kk) = dL(1)*ii*NL(nN,ii+1,jj,kk) +
    #                           dL(2)*jj*NL(nN,ii,jj+1,kk) +
    #                           dL(3)*kk*NL(nN,ii,jj,kk+1)   (one-based MATLAB)
    dnl_weighted = (
        d_l[0] * NLMatrix[:, 1:4, 0:3, 0:3] * ii_w[None, :, None, None]
        + d_l[1] * NLMatrix[:, 0:3, 1:4, 0:3] * ii_w[None, None, :, None]
        + d_l[2] * NLMatrix[:, 0:3, 0:3, 1:4] * ii_w[None, None, None, :]
    )
    return dnl_weighted


@_register("dxNL_matrix")
def dxNL_matrix(TriProps, NLMatrix):
    """Derivative expansion in x of the shape functions, port of dxNL_matrix.m."""
    return _dNL_matrix_weighted(TriProps.b, NLMatrix)


@_register("dyNL_matrix")
def dyNL_matrix(TriProps, NLMatrix):
    """Derivative expansion in y of the shape functions, port of dyNL_matrix.m."""
    return _dNL_matrix_weighted(TriProps.c, NLMatrix)


# ============================================================================
# Element matrices for the fluid (port of KM_el_matrix_fluid.m / KM_matrix_fluid.m)
# ============================================================================


@_register("KM_el_matrix_fluid")
def KM_el_matrix_fluid(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """Element integrand expansions for the fluid, port of KM_el_matrix_fluid.m."""
    n_nodes = int(CompStruct.Advanced.N_nodes)
    dvar = int(CompStruct.Data.DVarNum[ii_d])
    msize = dvar * n_nodes

    el_matrices = AttrDict()
    # Prepare various properties
    rho = np.asarray(ElPhysProps.RhoVec, dtype=float)
    rho2_lambda = np.asarray(ElPhysProps.Rho2LambdaVec, dtype=float)
    dxL = np.asarray(TriProps.dxL, dtype=float)
    dyL = np.asarray(TriProps.dyL, dtype=float)

    nnn = BasicMatrices.NNNConvMatrixInt  # (a, k, b)
    dnndn = BasicMatrices.dNNdNConvMatrixInt  # (i, j, a, k, b)

    # NtRho2LambdaN(a,b) = sum_k NNN(a,k,b) * Rho2Lambda(k)
    el_matrices.NtRho2LambdaNMatrix = np.tensordot(nnn, rho2_lambda, axes=([1], [0]))
    # B2tCB2(a,b) = sum_k NNN(a,k,b) * Rho(k)
    el_matrices.B2tCB2Matrix = np.tensordot(nnn, rho, axes=([1], [0]))
    el_matrices.B1tCB2Matrix = np.zeros((msize, msize), dtype=float)
    el_matrices.B2tCB1Matrix = np.zeros((msize, msize), dtype=float)
    # B1tCB1(a,b) = sum_{i,j,k} Rho(k)*(dxL(i)*dxL(j) + dyL(i)*dyL(j)) * dNNdN(i,j,a,k,b)
    # NB: the axes of dNNdNConvMatrixInt are (i, j, a, k, b) - the middle node
    # index k is axis 3, so the contraction is "ijakb", NOT "ijkab". Using
    # "ijkab" contracts Rho with the dN_a node index instead, which sums to
    # zero identically (sum_a dN_a/dL_i = d/dL_i(1) = 0) and nullifies K1.
    coeff = (np.outer(dxL, dxL) + np.outer(dyL, dyL))[:, :, None] * rho[None, None, :]
    el_matrices.B1tCB1Matrix = np.einsum("ijk,ijakb->ab", coeff, dnndn)
    return el_matrices


@_register("KM_matrix_fluid")
def KM_matrix_fluid(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """Element stiffness and mass matrices for the fluid, port of KM_matrix_fluid.m."""
    # get the triangle area
    delta = float(np.asarray(TriProps.delta, dtype=float).ravel()[0])

    el_matrices = KM_el_matrix_fluid(
        BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps
    )

    # mass matrix term - the kinetic energy
    el_matrices.MMatrix = delta * el_matrices.NtRho2LambdaNMatrix
    # stiffness matrix terms - the potential energy
    el_matrices.K1Matrix = (1.0 / delta) * el_matrices.B1tCB1Matrix
    el_matrices.K2Matrix = el_matrices.B1tCB2Matrix - el_matrices.B2tCB1Matrix
    el_matrices.K3Matrix = delta * el_matrices.B2tCB2Matrix
    return el_matrices


# ============================================================================
# Element matrices for the HTTI solid (port of KM_el_matrix_HTTI.m / KM_matrix_HTTI.m)
# ============================================================================


def _node_var_form(mat_ijab, n_nodes, dvar):
    """Convert a (i, j, a, b) tensor to the (a*dvar+i, b*dvar+j) matrix form.

    This is algebraically identical to the kron-based expansion into the
    'Large' matrices used in KM_el_matrix_HTTI.m.
    """
    return (
        mat_ijab.transpose(2, 0, 3, 1)
        .reshape(n_nodes * dvar, n_nodes * dvar)
    )


def _KM_el_matrix_HTTI_core(
    BasicMatrices,
    CompStruct,
    FEMatrices,
    ii_d,
    ii_el,
    ElPhysProps,
    TriProps,
    *,
    lx_by_node=None,
    ly_by_node=None,
    cij_scale=None,
    jacobian_scale=None,
):
    """Shared HTTI element kernel with optional PML/ABC node transforms.

    MATLAB's PML implementation replaces Lx/Ly at each interpolation node
    and multiplies both stiffness and density integrands by the complex
    coordinate Jacobian.  ABC instead scales Cij.  Keeping those transforms
    at node level lets the ordinary, PML, ABC and PML+ABC kernels share the
    same contraction code without changing the SAFE algebra.
    """
    n_nodes = int(CompStruct.Advanced.N_nodes)
    dvar = int(CompStruct.Data.DVarNum[ii_d])

    el_matrices = AttrDict()
    # Prepare various properties
    rho = np.asarray(ElPhysProps.RhoVec, dtype=float)
    cij_matrix = np.asarray(ElPhysProps.CijMatrix)  # (k, 6, 6)
    dxL = np.asarray(TriProps.dxL, dtype=float)
    dyL = np.asarray(TriProps.dyL, dtype=float)
    base_lx = np.asarray(BasicMatrices.Lx)
    base_ly = np.asarray(BasicMatrices.Ly)
    Lz = BasicMatrices.Lz

    if lx_by_node is None:
        lx_by_node = np.broadcast_to(base_lx, (n_nodes,) + base_lx.shape)
    else:
        lx_by_node = np.asarray(lx_by_node)
    if ly_by_node is None:
        ly_by_node = np.broadcast_to(base_ly, (n_nodes,) + base_ly.shape)
    else:
        ly_by_node = np.asarray(ly_by_node)
    if cij_scale is None:
        cij_scale = np.ones(n_nodes, dtype=float)
    cij_scale = np.asarray(cij_scale)
    if jacobian_scale is None:
        jacobian_scale = np.ones(n_nodes, dtype=float)
    jacobian_scale = np.asarray(jacobian_scale)

    cij_eff = cij_matrix * cij_scale[:, None, None]

    nnn = BasicMatrices.NNNConvMatrixInt  # (a, k, b)
    dnnn = BasicMatrices.dNNNConvMatrixInt  # (x, a, k, b)
    nndn = BasicMatrices.NNdNConvMatrixInt  # (x, a, k, b)
    dnndn = BasicMatrices.dNNdNConvMatrixInt  # (x, y, a, k, b)

    # MATLAB uses conj(Lx') rather than Lx'.  For complex PML Lx this is a
    # plain transpose (no Hermitian conjugation), which the einsums below
    # reproduce deliberately.
    LxTCijLx = np.einsum("kia,kij,kjb->kab", lx_by_node, cij_eff, lx_by_node)
    LxTCijLy = np.einsum("kia,kij,kjb->kab", lx_by_node, cij_eff, ly_by_node)
    LyTCijLx = np.einsum("kia,kij,kjb->kab", ly_by_node, cij_eff, lx_by_node)
    LyTCijLy = np.einsum("kia,kij,kjb->kab", ly_by_node, cij_eff, ly_by_node)
    LxTCijLz = np.einsum("kia,kij,jb->kab", lx_by_node, cij_eff, Lz)
    LzTCijLx = np.einsum("ia,kij,kjb->kab", Lz, cij_eff, lx_by_node)
    LyTCijLz = np.einsum("kia,kij,jb->kab", ly_by_node, cij_eff, Lz)
    LzTCijLy = np.einsum("ia,kij,kjb->kab", Lz, cij_eff, ly_by_node)
    LzTCijLz = np.einsum("ia,kij,jb->kab", Lz, cij_eff, Lz)

    jac = jacobian_scale[:, None, None]
    LxTCijLx = LxTCijLx * jac
    LxTCijLy = LxTCijLy * jac
    LyTCijLx = LyTCijLx * jac
    LyTCijLy = LyTCijLy * jac
    LxTCijLz = LxTCijLz * jac
    LzTCijLx = LzTCijLx * jac
    LyTCijLz = LyTCijLz * jac
    LzTCijLy = LzTCijLy * jac
    LzTCijLz = LzTCijLz * jac

    # B1tCB1(i,j,a,b) = sum_{x,y,k} M1(k,x,y,i,j) * dNNdN(x,y,a,k,b)
    m1 = (
        LxTCijLx[:, None, None, :, :] * (dxL[:, None] * dxL[None, :])[None, :, :, None, None]
        + LxTCijLy[:, None, None, :, :] * (dxL[:, None] * dyL[None, :])[None, :, :, None, None]
        + LyTCijLx[:, None, None, :, :] * (dyL[:, None] * dxL[None, :])[None, :, :, None, None]
        + LyTCijLy[:, None, None, :, :] * (dyL[:, None] * dyL[None, :])[None, :, :, None, None]
    )
    b1tcb1 = np.einsum("kxyij,xyakb->ijab", m1, dnndn)
    el_matrices.B1tCB1Matrix = _node_var_form(b1tcb1, n_nodes, dvar)

    # B1tCB2(i,j,a,b) = sum_{x,k} M2(k,x,i,j) * dNNN(x,a,k,b)
    m2 = (
        LxTCijLz[:, None, :, :] * dxL[None, :, None, None]
        + LyTCijLz[:, None, :, :] * dyL[None, :, None, None]
    )
    b1tcb2 = np.einsum("kxij,xakb->ijab", m2, dnnn)
    el_matrices.B1tCB2Matrix = _node_var_form(b1tcb2, n_nodes, dvar)

    # B2tCB1(i,j,a,b) = sum_{x,k} M3(k,x,i,j) * NNdN(x,a,k,b)
    m3 = (
        LzTCijLx[:, None, :, :] * dxL[None, :, None, None]
        + LzTCijLy[:, None, :, :] * dyL[None, :, None, None]
    )
    b2tcb1 = np.einsum("kxij,xakb->ijab", m3, nndn)
    el_matrices.B2tCB1Matrix = _node_var_form(b2tcb1, n_nodes, dvar)

    # B2tCB2(i,j,a,b) = sum_k LzTCijLz(k,i,j) * NNN(a,k,b)
    b2tcb2 = np.einsum("kij,akb->ijab", LzTCijLz, nnn)
    el_matrices.B2tCB2Matrix = _node_var_form(b2tcb2, n_nodes, dvar)

    # NtRhoN(i,j,a,b) = IdMatrix(i,j) * sum_k NNN(a,k,b) * Rho(k)
    s_ab = np.einsum("akb,k->ab", nnn, rho * jacobian_scale)
    nt_rho_n = np.zeros(
        (dvar, dvar, n_nodes, n_nodes),
        dtype=np.result_type(s_ab.dtype, cij_eff.dtype, lx_by_node.dtype, ly_by_node.dtype),
    )
    for ii in range(dvar):
        nt_rho_n[ii, ii] = s_ab
    el_matrices.NtRhoNMatrix = _node_var_form(nt_rho_n, n_nodes, dvar)
    return el_matrices


@_register("KM_el_matrix_HTTI")
def KM_el_matrix_HTTI(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """Element integrand expansions for the ordinary HTTI solid."""
    return _KM_el_matrix_HTTI_core(
        BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps
    )


def _abc_cij_scale(CompStruct, FEMatrices, ii_d, ii_el):
    """Return MATLAB-compatible per-node ABC stiffness multipliers.

    Two details intentionally mirror the supplied reference source exactly:
    (1) ABC_account_r is tested against numeric 1, so the literal string
        'yes' does not activate the 1/r factor in that MATLAB version;
    (2) CijMatrix is multiplied in-place as a whole on every node iteration,
        so the factor used at node k is the cumulative product through k.
    """
    n_nodes = int(CompStruct.Advanced.N_nodes)
    tri_nodes = FEMatrices.DElements[ii_d][0:10, ii_el].astype(int)
    xy = FEMatrices.MeshNodes[0:2, tri_nodes]
    rr = np.sqrt(xy[0] ** 2 + xy[1] ** 2)

    r_zv = float(CompStruct.Model.DomainRx[-2])
    layer = float(CompStruct.Model.DomainRx[-1]) - r_zv
    if layer <= 0.0:
        raise ValueError("ABC layer thickness must be positive")
    alpha = float(CompStruct.Model.ABC_factor)
    degree = float(CompStruct.Model.ABC_degree)
    sigma = alpha * (np.abs(rr - r_zv) / layer) ** degree

    account_r = CompStruct.Model.get("ABC_account_r", 0)
    if isinstance(account_r, (int, float, np.integer, np.floating)) and float(account_r) == 1.0:
        sigma = np.divide(sigma, rr, out=np.zeros_like(sigma), where=rr != 0.0)

    node_factor = 1.0 + 1j * sigma
    if ii_d == int(CompStruct.Data.N_domain) - 1:
        return np.cumprod(node_factor[:n_nodes])
    return np.ones(n_nodes, dtype=complex)


def _pml_node_transform(CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el):
    """Port the per-node coordinate stretch from KM_el_matrix_HTTI_PML.m."""
    n_nodes = int(CompStruct.Advanced.N_nodes)
    base_lx = np.asarray(BasicMatrices.Lx, dtype=complex)
    base_ly = np.asarray(BasicMatrices.Ly, dtype=complex)
    lx = np.broadcast_to(base_lx, (n_nodes,) + base_lx.shape).copy()
    ly = np.broadcast_to(base_ly, (n_nodes,) + base_ly.shape).copy()
    jac = np.ones(n_nodes, dtype=complex)

    if ii_d != int(CompStruct.Data.N_domain) - 1:
        return lx, ly, jac

    tri_nodes = FEMatrices.DElements[ii_d][0:10, ii_el].astype(int)
    xy_all = FEMatrices.MeshNodes[0:2, tri_nodes]
    method = int(round(float(CompStruct.Model.PML_method)))
    degree = float(CompStruct.Model.PML_degree)
    factor = float(CompStruct.Model.PML_factor)
    add_loc = str(CompStruct.Model.get("AddDomainLoc", "ext")).lower()

    for kk in range(n_nodes):
        x = float(xy_all[0, kk])
        y = float(xy_all[1, kk])

        if method == 1:
            r_node = float(np.hypot(x, y))
            if add_loc == "ext":
                r_zv = min(float(CompStruct.Model.DomainRx[-2]), float(CompStruct.Model.DomainRy[-2]))
                layer = min(float(CompStruct.Model.DomainRx[-1]), float(CompStruct.Model.DomainRy[-1])) - r_zv
            elif add_loc == "int":
                layer = float(CompStruct.Model.AddDomainL_m)
                r_zv = min(float(CompStruct.Model.DomainRx[-1]), float(CompStruct.Model.DomainRy[-1])) - layer
            else:
                raise ValueError(f"Unsupported AddDomainLoc: {add_loc}")
            if layer <= 0.0:
                raise ValueError("PML layer thickness must be positive")

            if r_node > r_zv:
                eta = (r_node - r_zv) / layer
                sigma_r = eta**degree
                gamma_r = 1j * factor * sigma_r
                r_tilde = r_node - 1j * (factor / (degree + 1.0)) * eta ** (degree + 1.0)
                var_xy = x * x / (gamma_r * r_node * r_node) + y * y / (r_tilde * r_node)
                var_yx = y * y / (gamma_r * r_node * r_node) + x * x / (r_tilde * r_node)
                var_all = (1.0 / (gamma_r * r_node * r_node) - 1.0 / (r_tilde * r_node)) * x * y
                lx[kk] = var_xy * base_lx + var_all * base_ly
                ly[kk] = var_all * base_lx + var_yx * base_ly
                jac[kk] = gamma_r * r_tilde / r_node

        elif method == 2:
            if add_loc == "ext":
                x_zv = float(CompStruct.Model.DomainRx[-2])
                lx_layer = float(CompStruct.Model.DomainRx[-1]) - x_zv
                y_zv = float(CompStruct.Model.DomainRy[-2])
                ly_layer = float(CompStruct.Model.DomainRy[-1]) - y_zv
            elif add_loc == "int":
                lx_layer = float(CompStruct.Model.AddDomainL_m)
                x_zv = float(CompStruct.Model.DomainRx[-1]) - lx_layer
                ly_layer = float(CompStruct.Model.AddDomainL_m)
                # The supplied MATLAB source uses DomainRx here (not DomainRy).
                y_zv = float(CompStruct.Model.DomainRx[-1]) - ly_layer
            else:
                raise ValueError(f"Unsupported AddDomainLoc: {add_loc}")
            if lx_layer <= 0.0 or ly_layer <= 0.0:
                raise ValueError("PML layer thickness must be positive")

            gamma_x = 1.0 + 0.0j
            if abs(x) > x_zv:
                sigma_x = ((abs(x) - x_zv) / lx_layer) ** degree
                gamma_x = 1.0 - 1j * factor * sigma_x
                lx[kk] = (1.0 / gamma_x) * base_lx

            gamma_y = 1.0 + 0.0j
            if abs(y) > y_zv:
                sigma_y = ((abs(y) - y_zv) / ly_layer) ** degree
                gamma_y = 1.0 - 1j * factor * sigma_y
                ly[kk] = (1.0 / gamma_y) * base_ly
            jac[kk] = gamma_x * gamma_y
        else:
            raise ValueError(f"Unsupported PML_method: {CompStruct.Model.PML_method}")

    return lx, ly, jac


@_register("KM_el_matrix_HTTI_ABC")
def KM_el_matrix_HTTI_ABC(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """MATLAB-compatible absorbing-boundary-condition HTTI element kernel."""
    scale = _abc_cij_scale(CompStruct, FEMatrices, ii_d, ii_el)
    return _KM_el_matrix_HTTI_core(
        BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps,
        cij_scale=scale,
    )


@_register("KM_el_matrix_HTTI_PML")
def KM_el_matrix_HTTI_PML(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """Port of KM_el_matrix_HTTI_PML.m."""
    lx, ly, jac = _pml_node_transform(CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el)
    return _KM_el_matrix_HTTI_core(
        BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps,
        lx_by_node=lx,
        ly_by_node=ly,
        jacobian_scale=jac,
    )


@_register("KM_el_matrix_HTTI_PML_ABC")
def KM_el_matrix_HTTI_PML_ABC(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """Combined PML+ABC kernel (composition of the two reference transforms)."""
    lx, ly, jac = _pml_node_transform(CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el)
    scale = _abc_cij_scale(CompStruct, FEMatrices, ii_d, ii_el)
    return _KM_el_matrix_HTTI_core(
        BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps,
        lx_by_node=lx,
        ly_by_node=ly,
        cij_scale=scale,
        jacobian_scale=jac,
    )


@_register("KM_matrix_HTTI")
def KM_matrix_HTTI(BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    """Element stiffness and mass matrices for the HTTI solid, port of KM_matrix_HTTI.m."""
    # get the triangle area
    delta = float(np.asarray(TriProps.delta, dtype=float).ravel()[0])

    el_matrices = call_method(
        CompStruct.Methods.KM_el_matrix[ii_d],
        BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps,
    )

    # mass matrix term - the kinetic energy
    el_matrices.MMatrix = -delta * el_matrices.NtRhoNMatrix
    # stiffness matrix terms - the potential energy
    el_matrices.K1Matrix = -(1.0 / delta) * el_matrices.B1tCB1Matrix
    el_matrices.K2Matrix = -(el_matrices.B1tCB2Matrix - el_matrices.B2tCB1Matrix)
    el_matrices.K3Matrix = -delta * el_matrices.B2tCB2Matrix
    return el_matrices


# ============================================================================
# Domain matrix assembly (port of MatricesParts_*_sp_SAFE_cubic.m)
# ============================================================================


def _tri_props(FEMatrices, ii_d, ii_el):
    """Read the element properties (a, b, c, delta as defined in Zienkiewicz)."""
    de_mesh_props = FEMatrices.DEMeshProps[ii_d]
    tri_props = AttrDict(
        a=de_mesh_props.a[:, ii_el],
        b=de_mesh_props.b[:, ii_el],
        c=de_mesh_props.c[:, ii_el],
        delta=np.atleast_1d(de_mesh_props.delta[ii_el]),
    )
    # Compute the vectors, which enter into the expressions for the matrices
    # of derivatives of the shape functions; d/dx L_i = 1/(2*delta) * b_i and
    # the factor 1/delta is omitted for the simplification of computations
    tri_props.dxL = 0.5 * tri_props.b
    tri_props.dyL = 0.5 * tri_props.c
    return tri_props


def _new_element_store(names):
    """Create the triplet store used to accumulate sparse domain matrices."""
    return {name: ([], [], []) for name in names}


def _accumulate_element(store, var_vec_arr, matrices):
    """Accumulate one element matrix into the triplet store.

    Ports the sparse insertion of MatricesParts_*_sp_SAFE_cubic.m:
    VarRows = repmat(VarVecArr,1,N), VarCols = reshape(repmat(VarVecArr,N,1),1,[])
    and the column-major reshape of the element matrix. Duplicate entries are
    summed, exactly as MATLAB sparse() does.
    """
    n_var = var_vec_arr.size
    var_rows = np.tile(var_vec_arr, n_var)
    var_cols = np.repeat(var_vec_arr, n_var)
    for name, mat in matrices.items():
        rows, cols, vals = store[name]
        rows.append(var_rows)
        cols.append(var_cols)
        # ABC/PML element matrices are complex.  Do not coerce them back to
        # float during sparse assembly or the attenuation/stretching terms
        # disappear silently.
        vals.append(np.asarray(mat).reshape(-1, order="F"))


def _store_to_sparse(store, size):
    """Convert the triplet store into CSR matrices (summing duplicates)."""
    out = {}
    for name, (rows, cols, vals) in store.items():
        out[name] = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(size, size),
        ).tocsr()
    return out


@_register("MatricesParts_fluid_sp_SAFE_cubic")
def MatricesParts_fluid_sp_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, ii_d):
    """Assemble domain matrix blocks for the fluid, port of MatricesParts_fluid_sp_SAFE_cubic.m."""
    d_elements = FEMatrices.DElements[ii_d]
    d_nodes = FEMatrices.DNodes[ii_d]
    n_domain_elements = d_elements.shape[1]
    dfe_var_num = d_nodes.size  # DVarNum == 1 for the fluid

    store = _new_element_store(("K1", "K2", "K3", "M"))

    # Process all of the elements, which belong to the domain
    for ii_el in range(n_domain_elements):
        # read the numbers of nodes, which belong to the element
        tri_nodes = d_elements[0:10, ii_el]
        # identify positions of the element nodes and the variables in the
        # respective matrices (DNodes is sorted, so searchsorted gives the
        # same positions as the MATLAB ismember/find loop)
        var_vec_arr = np.searchsorted(d_nodes, tri_nodes)

        # read physical properties of the element
        el_phys_props = call_method(
            CompStruct.Methods.getPhysProps[ii_d],
            CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el,
        )
        tri_props = _tri_props(FEMatrices, ii_d, ii_el)

        # compute the contributions of the element to the stiffness and mass matrices
        el_matrices = call_method(
            CompStruct.Methods.KM_matrix[ii_d],
            BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, el_phys_props, tri_props,
        )

        _accumulate_element(
            store,
            var_vec_arr,
            {
                "K1": el_matrices.K1Matrix,
                "K2": el_matrices.K2Matrix,
                "K3": el_matrices.K3Matrix,
                "M": el_matrices.MMatrix,
            },
        )

    blocks = _store_to_sparse(store, dfe_var_num)
    FEMatrices.K1Matrix_d[ii_d] = blocks["K1"]
    FEMatrices.K2Matrix_d[ii_d] = blocks["K2"]
    FEMatrices.K3Matrix_d[ii_d] = blocks["K3"]
    FEMatrices.MMatrix_d[ii_d] = blocks["M"]
    FEMatrices.PMatrix_d[ii_d] = sparse.csr_matrix((dfe_var_num, dfe_var_num))
    return FEMatrices


@_register("MatricesParts_HTTI_sp_SAFE_cubic")
def MatricesParts_HTTI_sp_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, ii_d):
    """Assemble domain matrix blocks for the HTTI solid, port of MatricesParts_HTTI_sp_SAFE_cubic.m."""
    d_elements = FEMatrices.DElements[ii_d]
    d_nodes = FEMatrices.DNodes[ii_d]
    dvar = int(CompStruct.Data.DVarNum[ii_d])
    n_domain_elements = d_elements.shape[1]
    dfe_var_num = dvar * d_nodes.size

    store = _new_element_store(("K1", "K2", "B1tCB2", "B2tCB1", "K3", "M"))

    # Process all of the elements, which belong to the domain
    for ii_el in range(n_domain_elements):
        # read the numbers of nodes, which belong to the element
        tri_nodes = d_elements[0:10, ii_el]
        # identify positions of the element nodes and the variables in the
        # respective matrices (node-major variable ordering, dvar per node)
        node_pos = np.searchsorted(d_nodes, tri_nodes)
        var_vec_arr = (dvar * node_pos[:, None] + np.arange(dvar)[None, :]).ravel()

        # read physical properties of the element
        el_phys_props = call_method(
            CompStruct.Methods.getPhysProps[ii_d],
            CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el,
        )
        tri_props = _tri_props(FEMatrices, ii_d, ii_el)

        # compute the contributions of the element to the stiffness and mass matrices
        el_matrices = call_method(
            CompStruct.Methods.KM_matrix[ii_d],
            BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, el_phys_props, tri_props,
        )

        _accumulate_element(
            store,
            var_vec_arr,
            {
                "K1": el_matrices.K1Matrix,
                "K2": el_matrices.K2Matrix,
                "B1tCB2": el_matrices.B1tCB2Matrix,
                "B2tCB1": el_matrices.B2tCB1Matrix,
                "K3": el_matrices.K3Matrix,
                "M": el_matrices.MMatrix,
            },
        )

    blocks = _store_to_sparse(store, dfe_var_num)
    FEMatrices.K1Matrix_d[ii_d] = blocks["K1"]
    FEMatrices.K2Matrix_d[ii_d] = blocks["K2"]
    FEMatrices.B1tCB2Matrix_d[ii_d] = blocks["B1tCB2"]
    FEMatrices.B2tCB1Matrix_d[ii_d] = blocks["B2tCB1"]
    FEMatrices.K3Matrix_d[ii_d] = blocks["K3"]
    FEMatrices.MMatrix_d[ii_d] = blocks["M"]
    FEMatrices.PMatrix_d[ii_d] = sparse.csr_matrix((dfe_var_num, dfe_var_num))
    return FEMatrices


# ============================================================================
# Interface condition matrices (port of ICMatrices_*_SAFE_cubic.m,
# IC_matrix_FS.m, IC_el_matrix_FS.m)
# ============================================================================


def _edge_nodes_pos(tri_nodes, node1, node2):
    """One-based positions of the four edge nodes inside the triangle.

    Ports the mid-side node identification block shared by the MATLAB
    ICMatrices_* and AssembleFullMatrices_* procedures. The returned
    positions are one-based, exactly as in the MATLAB code.
    """
    pos1 = int(np.nonzero(tri_nodes == node1)[0][0]) + 1
    pos2 = int(np.nonzero(tri_nodes == node2)[0][0]) + 1
    # find the interior nodes of the edge element
    if pos2 > pos1 % 3:
        pos3 = (pos1 + 1) * 2
        pos4 = (pos1 + 1) * 2 + 1
    else:
        pos3 = (pos2 + 1) * 2 + 1
        pos4 = (pos2 + 1) * 2
    return [pos1, pos2, pos3, pos4]


def _boundary_full_nodes(b_elements, d_elements):
    """Collect the four nodes (corners + mid-side) of every boundary edge."""
    b_nodes_full = []
    for ii_ed in range(b_elements.shape[1]):
        node1 = int(b_elements[0, ii_ed])
        node2 = int(b_elements[1, ii_ed])
        # find the element, which contains these nodes
        counts = np.isin(d_elements[0:10, :], (node1, node2)).sum(axis=0)
        edge_el = int(np.argmax(counts))
        tri_nodes = d_elements[0:10, edge_el]
        pos = _edge_nodes_pos(tri_nodes, node1, node2)
        b_nodes_full.extend((node1, node2, int(tri_nodes[pos[2] - 1]), int(tri_nodes[pos[3] - 1])))
    if not b_nodes_full:
        return np.zeros(0, dtype=int)
    return np.unique(np.asarray(b_nodes_full, dtype=int))


@_register("ICMatrices_ff_ss_SAFE_cubic")
def ICMatrices_ff_ss_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2):
    """Interface nodes for the ff/ss interface, port of ICMatrices_ff_ss_SAFE_cubic.m."""
    d1_var_num = int(CompStruct.Data.DVarNum[ii_d1]) * FEMatrices.DNodes[ii_d1].size
    d2_var_num = int(CompStruct.Data.DVarNum[ii_d2]) * FEMatrices.DNodes[ii_d2].size

    # create zero sparse matrix for the boundary condition matrix
    FEMatrices.ZeroD12 = sparse.csr_matrix((d1_var_num, d2_var_num))
    FEMatrices.ZeroD21 = sparse.csr_matrix((d1_var_num, d2_var_num))

    # prepare interface elements (boundary tags are one-based domain numbers)
    b_elements = FEMatrices.BoundaryEdges[0:2, FEMatrices.BoundaryEdges[2, :] == ii_int + 1]
    FEMatrices.BNodesFull[ii_int] = _boundary_full_nodes(b_elements, FEMatrices.DElements[ii_d1])
    return FEMatrices


@_register("IC_el_matrix_FS")
def IC_el_matrix_FS(BasicMatrices, CompStruct, ElPhysProps, EdgeProps, EdgeNodesPos):
    """Element FS interface condition matrix, port of IC_el_matrix_FS.m."""
    n_edge = int(CompStruct.Advanced.NEdge_nodes)

    # Prepare various properties (EdgeNodesPos.Df is one-based, as in MATLAB)
    rho = np.asarray(ElPhysProps.DfEl.RhoVec, dtype=float)[np.asarray(EdgeNodesPos.Df) - 1]
    normal = np.asarray(EdgeProps.Normal, dtype=float)

    nene = BasicMatrices.NENENEConvMatrixInt  # (a, k, n)

    # NfltRhonN(a, 3*n + i) = Normal(i) * sum_k NENENE(a,k,n) * Rho(k)
    # (algebraically identical to the kron-based 'Large' expansion of the
    # MATLAB code)
    s_an = np.einsum("akn,k->an", nene, rho)
    nflt_rho_n = np.zeros((n_edge, 3 * n_edge), dtype=float)
    for ii in range(3):
        nflt_rho_n[:, ii::3] = s_an * normal[ii]

    el_matrices = AttrDict()
    el_matrices.NfltRhonNMatrix = nflt_rho_n
    return el_matrices


@_register("IC_matrix_FS")
def IC_matrix_FS(BasicMatrices, CompStruct, ii_int, ii_df, ii_ds, ElPhysProps, EdgeProps, EdgeNodesPos):
    """Edge blocks of the FS interface matrix, port of IC_matrix_FS.m."""
    # get the length of the edge (the element of the boundary)
    d_l = float(EdgeProps.Dl)

    el_matrices = call_method(
        CompStruct.Methods.IC_el_matrix[ii_int],
        BasicMatrices, CompStruct, ElPhysProps, EdgeProps, EdgeNodesPos,
    )

    b_matrix_dfs_edge = d_l * el_matrices.NfltRhonNMatrix
    b_matrix_dsf_edge = d_l * el_matrices.NfltRhonNMatrix.T
    return b_matrix_dfs_edge, b_matrix_dsf_edge


@_register("ICMatrices_fluid_HTTI_SAFE_cubic")
def ICMatrices_fluid_HTTI_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2):
    """Interface matrices for the fluid-HTTI interface, port of ICMatrices_fluid_HTTI_SAFE_cubic.m."""
    domain_type = CompStruct.Model.DomainType
    d1_var_num = int(CompStruct.Data.DVarNum[ii_d1]) * FEMatrices.DNodes[ii_d1].size
    d2_var_num = int(CompStruct.Data.DVarNum[ii_d2]) * FEMatrices.DNodes[ii_d2].size

    # create zero sparse matrix for the fluid-solid boundary condition matrix
    FEMatrices.ZeroD12 = sparse.csr_matrix((d1_var_num, d2_var_num))
    FEMatrices.ZeroD21 = sparse.csr_matrix((d2_var_num, d1_var_num))

    # identify, what is the type of the first domain - fluid or solid;
    # put fluid domain first for the computation of the IC matrices and
    # later adjust accordingly
    if domain_type[ii_d1] == "fluid":
        ii_df, ii_ds = ii_d1, ii_d2
    else:
        ii_df, ii_ds = ii_d2, ii_d1

    # boundary elements in 2d formulation are edges
    b_elements = FEMatrices.BoundaryEdges[0:2, FEMatrices.BoundaryEdges[2, :] == ii_int + 1]

    df_nodes = FEMatrices.DNodes[ii_df]
    ds_nodes = FEMatrices.DNodes[ii_ds]
    dvar_f = int(CompStruct.Data.DVarNum[ii_df])
    dvar_s = int(CompStruct.Data.DVarNum[ii_ds])
    bfe_var_num_fluid = dvar_f * df_nodes.size
    bfe_var_num_solid = dvar_s * ds_nodes.size

    store = _new_element_store(("Bfs",))
    b_nodes_full = []

    # Process all of the edges, which belong to the boundary
    for ii_ed in range(b_elements.shape[1]):
        # find the start and the end nodes of the edge; depending on whether
        # the fluid domain is the first or the last, the orientation of the
        # edge changes (the normal should be directed from the fluid into
        # the solid)
        if domain_type[ii_d1] == "fluid":
            node1 = int(b_elements[0, ii_ed])
            node2 = int(b_elements[1, ii_ed])
        else:
            node1 = int(b_elements[1, ii_ed])
            node2 = int(b_elements[0, ii_ed])

        # find the elements, which contain these nodes
        counts_f = np.isin(FEMatrices.DElements[ii_df][0:10, :], (node1, node2)).sum(axis=0)
        df_edge_el = int(np.argmax(counts_f))
        counts_s = np.isin(FEMatrices.DElements[ii_ds][0:10, :], (node1, node2)).sum(axis=0)
        ds_edge_el = int(np.argmax(counts_s))

        # read the nodes, which belong to the adjoint (to the edge) elements
        tri_nodes_f = FEMatrices.DElements[ii_df][0:10, df_edge_el]
        tri_nodes_s = FEMatrices.DElements[ii_ds][0:10, ds_edge_el]

        # find the nodes in between (mid-side nodes for cubic interpolation)
        pos_f = _edge_nodes_pos(tri_nodes_f, node1, node2)
        edge_nodes = [node1, node2, int(tri_nodes_f[pos_f[2] - 1]), int(tri_nodes_f[pos_f[3] - 1])]
        pos_s = [int(np.nonzero(tri_nodes_s == node)[0][0]) + 1 for node in edge_nodes]
        edge_nodes_pos = AttrDict(Df=np.asarray(pos_f), Ds=np.asarray(pos_s))

        b_nodes_full.extend(edge_nodes)

        # find the components of the edge vector
        edge_x = FEMatrices.MeshNodes[0, [node1, node2]]
        edge_y = FEMatrices.MeshNodes[1, [node1, node2]]
        edge_props = AttrDict()
        edge_props.DxEdge = float(edge_x[1] - edge_x[0])
        edge_props.DyEdge = float(edge_y[1] - edge_y[0])
        # length of the boundary element
        edge_props.Dl = float(np.hypot(edge_props.DxEdge, edge_props.DyEdge))
        # define the oriented normal to the boundary edge by taking the cross
        # product of the edge vector with the (-e_z) basis vector
        edge_props.Normal = np.cross(
            [edge_props.DxEdge / edge_props.Dl, edge_props.DyEdge / edge_props.Dl, 0.0],
            [0.0, 0.0, -1.0],
        )

        # read physical properties of the elements and prepare the matrices
        # for the expansion of these elements into interpolating functions L_i
        el_phys_props = AttrDict()
        el_phys_props.DfEl = call_method(
            CompStruct.Methods.getPhysProps[ii_df],
            CompStruct, BasicMatrices, FEMatrices, ii_df, df_edge_el,
        )
        el_phys_props.DsEl = call_method(
            CompStruct.Methods.getPhysProps[ii_ds],
            CompStruct, BasicMatrices, FEMatrices, ii_ds, ds_edge_el,
        )

        # compute the contributions of the element to the interface matrices
        b_matrix_dfs_edge, _b_matrix_dsf_edge = call_method(
            CompStruct.Methods.IC_matrix[ii_int],
            BasicMatrices, CompStruct, ii_int, ii_df, ii_ds,
            el_phys_props, edge_props, edge_nodes_pos,
        )

        # identify positions of the element nodes and the variables in the
        # respective matrices
        var_row_arr = []
        var_col_arr = []
        for node in edge_nodes:
            df_node_pos = int(np.searchsorted(df_nodes, node))
            ds_node_pos = int(np.searchsorted(ds_nodes, node))
            var_row_arr.extend(dvar_f * df_node_pos + np.arange(dvar_f))
            var_col_arr.extend(dvar_s * ds_node_pos + np.arange(dvar_s))
        # the row/column sizes differ from the square element case, so the
        # insertion is done explicitly (same repmat pattern as in MATLAB)
        var_row_arr = np.asarray(var_row_arr, dtype=int)
        var_col_arr = np.asarray(var_col_arr, dtype=int)
        rows, cols, vals = store["Bfs"]
        rows.append(np.tile(var_row_arr, var_col_arr.size))
        cols.append(np.repeat(var_col_arr, var_row_arr.size))
        vals.append(np.asarray(b_matrix_dfs_edge, dtype=float).reshape(-1, order="F"))

    rows, cols, vals = store["Bfs"]
    b_matrix_dfs = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(bfe_var_num_fluid, bfe_var_num_solid),
    ).tocsr()

    # Assemble the full list of the nodes, which belong to the interface
    FEMatrices.BNodesFull[ii_int] = np.unique(np.asarray(b_nodes_full, dtype=int))

    # Compute the solid-fluid boundary condition matrix - it is transpose of
    # the fluid-solid one according to the formulation
    b_matrix_dsf = b_matrix_dfs.T

    # assign the rectangular blocks of the matrices responsible for the
    # boundary conditions according to which domain is the first - solid or
    # fluid
    if domain_type[ii_d1] == "fluid":
        FEMatrices.PMatrixD12[ii_int] = b_matrix_dfs  # ++ or -- is Ok
        FEMatrices.PMatrixD21[ii_int] = b_matrix_dsf
    else:
        # change the sign because the normal vector should be directed from
        # the fluid into the solid
        FEMatrices.PMatrixD12[ii_int] = -b_matrix_dsf
        FEMatrices.PMatrixD21[ii_int] = -b_matrix_dfs
    return FEMatrices


# ============================================================================
# Block positions (port of BasicMatrices/FindPos_sp_SAFE.m)
# ============================================================================


@_register("FindPos_sp_SAFE")
def FindPos_sp_SAFE(CompStruct, FEMatrices, BasicMatrices):
    """Compute start positions of the domain blocks, port of FindPos_sp_SAFE.m.

    The positions are stored as zero-based half-open ranges [beg, end),
    unlike the one-based inclusive ranges of the MATLAB code.
    """
    pos_end = 0
    node_pos_end = 0
    BasicMatrices.Pos = []
    BasicMatrices.NodePos = []
    for ii_d in range(int(CompStruct.Data.N_domain)):
        pos_beg = pos_end
        node_pos_beg = node_pos_end
        num_nodes = FEMatrices.DNodes[ii_d].size
        pos_end = pos_beg + int(CompStruct.Data.DVarNum[ii_d]) * num_nodes
        node_pos_end = node_pos_beg + num_nodes
        BasicMatrices.Pos.append((pos_beg, pos_end))
        BasicMatrices.NodePos.append((node_pos_beg, node_pos_end))
    return BasicMatrices


# ============================================================================
# Full matrix assembly (port of AssembleFullMatrices_*_SAFE_cubic.m)
# ============================================================================


def _domain_var_positions(BasicMatrices, CompStruct, FEMatrices, ii_d, nodes):
    """Positions of the variables of the given nodes inside the full vector."""
    dvar = int(CompStruct.Data.DVarNum[ii_d])
    node_pos = np.searchsorted(FEMatrices.DNodes[ii_d], np.asarray(nodes, dtype=int))
    pos_beg = BasicMatrices.Pos[ii_d][0]
    return pos_beg + (dvar * node_pos[:, None] + np.arange(dvar)[None, :]).ravel()


def _sum_and_zero(mat, add_var_pos, remove_var_pos):
    """Sum rows/columns of coincident nodes and zero the removed ones.

    Ports the row/column operations of AssembleFullMatrices_ff_ss_SAFE_cubic.m:
    first the rows and columns of RemoveVarPos are added to AddVarPos, then
    the RemoveVarPos rows and columns are set to zero.
    """
    lil = mat.tolil()
    lil[add_var_pos, :] = lil[add_var_pos, :] + lil[remove_var_pos, :]
    lil[:, add_var_pos] = lil[:, add_var_pos] + lil[:, remove_var_pos]
    lil[remove_var_pos, :] = 0
    lil[:, remove_var_pos] = 0
    return lil.tocsr()


@_register("AssembleFullMatrices_ff_ss_SAFE_cubic")
def AssembleFullMatrices_ff_ss_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):
    """Merge domain blocks at the ff/ss interface, port of AssembleFullMatrices_ff_ss_SAFE_cubic.m."""
    # adding the matrices for the next block
    for name in ("K1", "K2", "K3", "M", "P"):
        full_name = f"{name}Matrix"
        FullMatrices[full_name] = sparse.block_diag(
            (FullMatrices[full_name], FEMatrices[f"{name}Matrix_d"][ii_d2]), format="csr"
        )

    # Adjust the list of variables (remove duplicates, if any, etc.)
    b_nodes_full = FEMatrices.BNodesFull[ii_int]
    FEMatrices.DNodesRem[ii_d2] = np.union1d(FEMatrices.DNodesRem[ii_d2], b_nodes_full)
    remove_nodes_mask = np.isin(FEMatrices.DNodes[ii_d2], FEMatrices.DNodesRem[ii_d2])
    FEMatrices.DNodesComp[ii_d2] = FEMatrices.DNodes[ii_d2][~remove_nodes_mask]

    # finding positions of the rows, which correspond to the nodes,
    # which will be removed (the values are taken from the coincident
    # nodes of the first domain)
    add_var_pos = _domain_var_positions(BasicMatrices, CompStruct, FEMatrices, ii_d1, b_nodes_full)
    remove_var_pos = _domain_var_positions(BasicMatrices, CompStruct, FEMatrices, ii_d2, b_nodes_full)

    FEMatrices.DTakeFromVarPos[ii_d2] = np.concatenate(
        (FEMatrices.DTakeFromVarPos[ii_d2], add_var_pos)
    ).astype(int)
    FEMatrices.DPutToVarPos[ii_d2] = np.concatenate(
        (FEMatrices.DPutToVarPos[ii_d2], remove_var_pos)
    ).astype(int)

    # Sum the rows and columns, which correspond to the coincident nodes
    # in both domains, then put to zero the elements, which will be removed
    for name in ("K1", "K2", "K3", "M", "P"):
        full_name = f"{name}Matrix"
        FullMatrices[full_name] = _sum_and_zero(FullMatrices[full_name], add_var_pos, remove_var_pos)
    return FEMatrices, FullMatrices


@_register("AssembleFullMatrices_fs_SAFE_cubic")
def AssembleFullMatrices_fs_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):
    """Merge domain blocks at the fluid-solid interface, port of AssembleFullMatrices_fs_SAFE_cubic.m."""
    # adding the matrices for the next block
    for name in ("K1", "K2", "K3", "M"):
        full_name = f"{name}Matrix"
        FullMatrices[full_name] = sparse.block_diag(
            (FullMatrices[full_name], FEMatrices[f"{name}Matrix_d"][ii_d2]), format="csr"
        )

    # the PMatrix also contains the rectangular interface blocks
    p_matrix = FullMatrices.PMatrix
    p_matrix_d12 = FEMatrices.PMatrixD12[ii_int]
    p_matrix_d21 = FEMatrices.PMatrixD21[ii_int]
    zero_matrix12 = sparse.csr_matrix((p_matrix.shape[0] - p_matrix_d12.shape[0], p_matrix_d12.shape[1]))
    insert_p_matrix_d12 = sparse.vstack([zero_matrix12, p_matrix_d12], format="csr")
    zero_matrix21 = sparse.csr_matrix((p_matrix_d21.shape[0], p_matrix.shape[1] - p_matrix_d21.shape[1]))
    insert_p_matrix_d21 = sparse.hstack([zero_matrix21, p_matrix_d21], format="csr")
    FullMatrices.PMatrix = sparse.bmat(
        [[p_matrix, insert_p_matrix_d12], [insert_p_matrix_d21, FEMatrices.PMatrix_d[ii_d2]]],
        format="csr",
    )

    # no nodes are removed from the computation at the FS interface
    FEMatrices.DNodesRem[ii_d2] = np.unique(FEMatrices.DNodesRem[ii_d2])
    remove_nodes_mask = np.isin(FEMatrices.DNodes[ii_d2], FEMatrices.DNodesRem[ii_d2])
    FEMatrices.DNodesComp[ii_d2] = FEMatrices.DNodes[ii_d2][~remove_nodes_mask]
    return FEMatrices, FullMatrices


@_register("AssembleFullMatrices_rigid_SAFE_cubic")
def AssembleFullMatrices_rigid_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):
    """Apply the rigid outer boundary, port of AssembleFullMatrices_rigid_SAFE_cubic.m.

    Rigid means that the corresponding displacements are zero; the removal
    of the rows and columns is done in RemoveRedundantVariables_SAFE.
    """
    # boundary elements in 2d formulation are edges
    b_elements = FEMatrices.BoundaryEdges[0:2, FEMatrices.BoundaryEdges[2, :] == ii_int + 1]

    # find all of the nodes, which belong to the outer boundary and which
    # should be removed from the computation and the full matrices
    b_nodes_full = _boundary_full_nodes(b_elements, FEMatrices.DElements[ii_d1])
    FEMatrices.BNodesFull[ii_int] = b_nodes_full

    # identify nodes, which should be removed from the computation
    FEMatrices.DNodesRem[ii_d1] = np.union1d(FEMatrices.DNodesRem[ii_d1], b_nodes_full)
    remove_nodes_mask = np.isin(FEMatrices.DNodes[ii_d1], FEMatrices.DNodesRem[ii_d1])
    FEMatrices.DNodesComp[ii_d1] = FEMatrices.DNodes[ii_d1][~remove_nodes_mask]

    # identify nodes and variables, which should be substituted by zero in
    # the full matrices after computing the eigenvectors of the reduced
    # matrices
    remove_var_pos = _domain_var_positions(BasicMatrices, CompStruct, FEMatrices, ii_d1, b_nodes_full)
    FEMatrices.DZeroVarPos[ii_d1] = np.concatenate(
        (FEMatrices.DZeroVarPos[ii_d1], remove_var_pos)
    ).astype(int)
    return FEMatrices, FullMatrices


@_register("AssembleFullMatrices_free_SAFE_cubic")
def AssembleFullMatrices_free_SAFE_cubic(CompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):
    """Apply the free outer boundary, port of AssembleFullMatrices_free_SAFE_cubic.m.

    No nodes are removed; only the boundary node list is recorded.
    """
    # boundary elements in 2d formulation are edges
    b_elements = FEMatrices.BoundaryEdges[0:2, FEMatrices.BoundaryEdges[2, :] == ii_int + 1]
    FEMatrices.BNodesFull[ii_int] = _boundary_full_nodes(b_elements, FEMatrices.DElements[ii_d1])
    remove_nodes_mask = np.isin(FEMatrices.DNodes[ii_d1], FEMatrices.DNodesRem[ii_d1])
    FEMatrices.DNodesComp[ii_d1] = FEMatrices.DNodes[ii_d1][~remove_nodes_mask]
    return FEMatrices, FullMatrices


# ============================================================================
# Removal of redundant variables (port of RemoveRedundantVariables_SAFE.m)
# ============================================================================


@_register("RemoveRedundantVariables_SAFE")
def RemoveRedundantVariables_SAFE(CompStruct, BasicMatrices, FEMatrices, FullMatrices):
    """Remove rows/columns of the removed nodes, port of RemoveRedundantVariables_SAFE.m."""
    n_domain = int(CompStruct.Data.N_domain)
    full_var_arr = np.arange(BasicMatrices.Pos[n_domain - 1][1])

    # Assemble array of the variables, which should be removed from
    # the problem (NB! Nodes in DNodes and DNodesRem are ordered!)
    full_remove_var_arr = []
    for ii_d in range(n_domain):
        d_nodes_rem = np.asarray(FEMatrices.DNodesRem[ii_d], dtype=int)
        if d_nodes_rem.size:
            remove_nodes_pos = np.searchsorted(FEMatrices.DNodes[ii_d], d_nodes_rem)
            dvar = int(CompStruct.Data.DVarNum[ii_d])
            d_remove_var_pos = BasicMatrices.Pos[ii_d][0] + (
                dvar * remove_nodes_pos[:, None] + np.arange(dvar)[None, :]
            ).ravel()
            full_remove_var_arr.append(d_remove_var_pos)
    if full_remove_var_arr:
        full_remove_var_arr = np.concatenate(full_remove_var_arr)
    else:
        full_remove_var_arr = np.zeros(0, dtype=int)

    full_keep_var_arr = np.setdiff1d(full_var_arr, full_remove_var_arr)
    # kept for the reconstruction of the full eigenvectors in St52
    FEMatrices.FullKeepVarArr = full_keep_var_arr

    # Remove the rows and columns, which correspond to the removed nodes
    for name in ("K1", "K2", "K3", "M", "P"):
        full_name = f"{name}Matrix"
        mat = FullMatrices[full_name].tocsr()
        FullMatrices[full_name] = mat[full_keep_var_arr, :][:, full_keep_var_arr]
    return FEMatrices, FullMatrices
