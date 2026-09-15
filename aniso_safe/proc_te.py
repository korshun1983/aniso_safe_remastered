"""TE postprocessing chain.

Ports of proc_aniso_TE.m, spectrum/St61_Proc_Fluid_TE_sp_SAFE.m and
spectrum/St61_Proc_Solid_TE_sp_SAFE.m.

The driver loads CompStruct.npz / FEMatrices-<freq>.npz / Results-<freq>.npz
produced by gen_aniso, builds a polar (r, theta) grid inside every layer,
interpolates the eigenvector displacements onto it, computes the radial
density of the kinetic energy and the trigonometric Fourier decomposition
of (ur, uf, uz). Results are stored in LayerTE-<freq>-<layer>.npz and
Slowness.npz files (numpy counterparts of the MATLAB .mat files).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .structures import AttrDict, as_attrdict


# ============================================================================
# Loading helpers
# ============================================================================
def resolve_data_dir(dir_name):
    """Return the directory that actually holds the gen_aniso output files.

    Accepts the output directory itself or its parent (with the data one
    level below, e.g. from an older layout that used output/ or results/).
    """
    path = Path(dir_name)
    if (path / "CompStruct.npz").exists():
        return path
    for sub in ("output", "results"):
        if (path / sub / "CompStruct.npz").exists():
            print(f"\tUsing data directory: {path / sub}")
            return path / sub
    return path


def load_compstruct(dir_name):
    """Restore CompStruct from the pickled plain dict inside CompStruct.npz."""
    comp_file = Path(dir_name) / "CompStruct.npz"
    if not comp_file.exists():
        raise FileNotFoundError(f"File is not existed! {comp_file}")
    with np.load(comp_file, allow_pickle=False) as data:
        return as_attrdict(pickle.loads(data["CompStruct"].tobytes()))


def load_fematrices(path):
    """Restore the FEMatrices structure saved by gen_aniso (per-domain lists)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File is not existed! {path}")
    with np.load(path, allow_pickle=False) as data:
        n_domain = int(data["N_domain"][0])
        fem = AttrDict(
            MeshNodes=data["MeshNodes"],
            BoundaryEdges=data["BoundaryEdges"],
            MeshTri=data["MeshTri"],
            MeshProps=AttrDict(delta=data["delta"], a=data["a"], b=data["b"], c=data["c"]),
            DomainRx=data["DomainRx"],
            DomainRy=data["DomainRy"],
        )
        fem.PhysProp = []
        fem.DElements = []
        fem.DEMeshProps = []
        fem.DNodes = []
        fem.DNodesRem = []
        fem.DNodesComp = []
        fem.DTakeFromVarPos = []
        fem.DPutToVarPos = []
        fem.DZeroVarPos = []
        for ii_d in range(n_domain):
            fem.DElements.append(data[f"DElements_{ii_d}"])
            fem.DEMeshProps.append(
                AttrDict(
                    delta=data[f"DEMeshProps_delta_{ii_d}"],
                    a=data[f"DEMeshProps_a_{ii_d}"],
                    b=data[f"DEMeshProps_b_{ii_d}"],
                    c=data[f"DEMeshProps_c_{ii_d}"],
                )
            )
            fem.DNodes.append(data[f"DNodes_{ii_d}"])
            fem.DNodesRem.append(data[f"DNodesRem_{ii_d}"])
            fem.DNodesComp.append(data[f"DNodesComp_{ii_d}"])
            fem.DTakeFromVarPos.append(data[f"DTakeFromVarPos_{ii_d}"])
            fem.DPutToVarPos.append(data[f"DPutToVarPos_{ii_d}"])
            fem.DZeroVarPos.append(data[f"DZeroVarPos_{ii_d}"])
            fem.PhysProp.append(as_attrdict(pickle.loads(data[f"PhysProp_{ii_d}"].tobytes())))
        fem.BNodes = [data[key] for key in data.files if key.startswith("BNodes_")]
        fem.BNodesFull = [data[key] for key in data.files if key.startswith("BNodesFull_")]
    return fem


def load_results(path):
    """Restore the eigenvalue results saved by gen_aniso."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File is not existed! {path}")
    with np.load(path, allow_pickle=False) as data:
        return AttrDict(
            REig_vals=data["eigenvalues"],
            REig_vecs=data["eigenvectors"],
            omega_val=float(data["omega_val"][0]),
        )


def _pol2cart(theta, r):
    """Port of pol2cart."""
    return r * np.cos(theta), r * np.sin(theta)


def _scattered(xx, yy, values):
    """Linear scattered interpolant, port role of scatteredInterpolant/TriScatteredInterp.

    LinearNDInterpolator also returns NaN outside the convex hull of the
    points, which matches the MATLAB default ('linear', no extrapolation).
    """
    return LinearNDInterpolator(np.column_stack([np.asarray(xx, float), np.asarray(yy, float)]), values)


def _corner_ind_rr(rr_var):
    """Node-position table for the rr-grid walk (0-based positions).

    Port of the ind_rr_1/ind_rr_2/ind_rr selection in St61_*_TE_sp_SAFE.m.
    Returns the positions for the three corner cases (corner 1, 2, 3).
    """
    if rr_var[1] > rr_var[2]:
        corner1 = (8, 7, 2)  # MATLAB 9, 8, 3
    else:
        corner1 = (3, 4, 1)  # MATLAB 4, 5, 2
    if rr_var[0] > rr_var[2]:
        corner2 = (5, 6, 2)  # MATLAB 6, 7, 3
    else:
        corner2 = (4, 3, 0)  # MATLAB 5, 4, 1
    if rr_var[0] > rr_var[1]:
        corner3 = (6, 5, 1)  # MATLAB 7, 6, 2
    else:
        corner3 = (7, 8, 0)  # MATLAB 8, 9, 1
    return corner1, corner2, corner3


# ============================================================================
# Fluid layer processing, port of St61_Proc_Fluid_TE_sp_SAFE.m
# ============================================================================
def St61_Proc_Fluid_TE_sp_SAFE(CompStructA, FEMatricesA, ProcTE, Results):
    """Process kinetic energy of a fluid layer for one frequency and all eigs."""
    ii_l = ProcTE.ilayer  # 0-based layer index (MATLAB ii_l is 1-based)
    cone = 1j
    n_nodes = int(CompStructA.Advanced.N_nodes)
    domain_tag = ii_l + 1  # domain tags in MeshTri stay 1-based

    # ========================================================================
    # Build Polar Grid
    # ========================================================================
    rr = []
    rr_min_prev_var = float(FEMatricesA.DomainRx[ii_l])

    # find fluid triangles and their nodes
    tri_fluid_ind = FEMatricesA.MeshTri[n_nodes, :] == domain_tag
    tri_fluid = FEMatricesA.MeshTri[0:n_nodes, tri_fluid_ind].astype(int).copy()
    if ii_l == 0:  # the smallest r of mesh
        corner_nodes = tri_fluid[0:3, :]
        rr_min_mesh = float(
            np.min(
                np.sqrt(
                    FEMatricesA.MeshNodes[0, corner_nodes] ** 2
                    + FEMatricesA.MeshNodes[1, corner_nodes] ** 2
                )
            )
        )
    else:
        rr_min_mesh = float(FEMatricesA.DomainRx[ii_l - 1])

    # find all triangles lying on external circle
    rr_circ = []
    for iel in range(tri_fluid.shape[1]):
        xx = FEMatricesA.MeshNodes[0, tri_fluid[:, iel]]
        yy = FEMatricesA.MeshNodes[1, tri_fluid[:, iel]]
        rr2 = (xx / FEMatricesA.DomainRx[ii_l]) ** 2 + (yy / FEMatricesA.DomainRy[ii_l]) ** 2
        rr2_sorted = np.sort(rr2)[::-1]
        if rr2_sorted[0] > 0.99 and rr2_sorted[1] > 0.99:
            rr_circ.append(float(np.sqrt(xx[9] ** 2 + yy[9] ** 2)))

    # set initial node with r=r_max and z=0
    r_find = float(FEMatricesA.DomainRx[ii_l])

    r_end_corr = 1.0
    max_walk = 4 * tri_fluid.shape[1] + 10  # safety guard against infinite loops
    i_count = 0
    while True:
        i_count += 1
        if i_count > max_walk:
            raise RuntimeError("rr-grid walk in the fluid layer did not terminate")
        node_candidates = np.nonzero(FEMatricesA.MeshNodes[0, :] == r_find)[0]
        if node_candidates.size == 0:
            break
        node_find = int(node_candidates[0])  # MATLAB keeps only the first match

        # find numbers of triangles containing node_find
        num_tri_node_find = np.nonzero((tri_fluid == node_find).any(axis=0))[0]
        if num_tri_node_find.size == 0:
            break
        tri_nodes = tri_fluid[:, num_tri_node_find]

        rr_min_var = rr_min_prev_var
        tri_node_min = 0
        for iet in range(tri_nodes.shape[1]):
            tri_node_xx = FEMatricesA.MeshNodes[0, tri_nodes[:, iet]]
            tri_node_yy = FEMatricesA.MeshNodes[1, tri_nodes[:, iet]]
            rr_var3 = np.sqrt(tri_node_xx[0:3] ** 2 + tri_node_yy[0:3] ** 2)  # nodes 1,2,3 only
            for ia in range(3):
                if rr_var3[ia] < rr_min_var:
                    rr_min_var = rr_var3[ia]
                    tri_node_min = iet

        tri_node_xx = FEMatricesA.MeshNodes[0, tri_nodes[:, tri_node_min]]
        tri_node_yy = FEMatricesA.MeshNodes[1, tri_nodes[:, tri_node_min]]
        rr_var = np.sqrt(tri_node_xx**2 + tri_node_yy**2)
        corner1, corner2, corner3 = _corner_ind_rr(rr_var)
        if tri_nodes[0, tri_node_min] == node_find:  # 1 node
            ind_rr_1, ind_rr_2, ind_rr = corner1
        elif tri_nodes[1, tri_node_min] == node_find:  # 2 node
            ind_rr_1, ind_rr_2, ind_rr = corner2
        else:  # 3 node
            ind_rr_1, ind_rr_2, ind_rr = corner3

        for iet in range(tri_nodes.shape[1]):  # choose 10 node
            node10 = tri_nodes[n_nodes - 1, iet]
            rr.append(
                float(np.sqrt(FEMatricesA.MeshNodes[0, node10] ** 2 + FEMatricesA.MeshNodes[1, node10] ** 2))
            )
        rr_min_prev_var = rr_var[ind_rr]

        if rr_min_prev_var <= rr_min_mesh * 1.01:
            break

        r_find = float(tri_node_xx[ind_rr])
        tri_fluid = np.delete(tri_fluid, num_tri_node_find, axis=1)

    rr_circ_min = min(rr_circ) * 0.995
    rr = np.asarray([r for r in rr if r <= rr_circ_min])
    rr = np.unique(np.sort(rr))
    rr = np.append(rr, rr_circ_min)

    # tuning of rr(1) and rr(end)
    nrr = len(rr)
    rr_ind = np.zeros(nrr, dtype=bool)
    for irr in range(nrr - 1, 0, -1):
        if abs(rr[irr] - rr[irr - 1]) / rr[irr] < 0.03:  # 3%
            rr_ind[irr] = True
    rr = rr[~rr_ind]
    rr[0] = rr[0] * 1.02
    rr[-1] = rr[-1] * r_end_corr
    nrr = len(rr)

    ResFluid = AttrDict()
    ResFluid.rr = rr

    # ========================================================================
    rho_fluid = float(FEMatricesA.PhysProp[ii_l].rho) * 1000.0
    dtheta_v = np.pi / (int(CompStructA.Model.DomainNth[ii_l]) * ProcTE.DThetaInc)
    theta_v = np.arange(-np.pi, np.pi + 0.5 * dtheta_v, dtheta_v)
    n_theta_v = len(theta_v)
    # end of building of polar grid
    # ========================================================================

    # make memory for answer
    n_eigs = len(ProcTE.neigs)
    ResFluid.TE_fer = np.zeros((n_eigs, nrr))  # T Energy
    # last index: 0 - for plus, 1 - for minus
    ResFluid.cF_ur_rm_pm = np.zeros((n_eigs, nrr, ProcTE.NHarm, 2), dtype=complex)
    ResFluid.cF_uf_rm_pm = np.zeros((n_eigs, nrr, ProcTE.NHarm, 2), dtype=complex)
    ResFluid.cF_uz_rm_pm = np.zeros((n_eigs, nrr, ProcTE.NHarm, 2), dtype=complex)

    # Common parameters for all freqs
    # Rectangular grid
    xx = FEMatricesA.MeshNodes[0, FEMatricesA.DNodesComp[ii_l]]
    yy = FEMatricesA.MeshNodes[1, FEMatricesA.DNodesComp[ii_l]]

    ind_tria_fluid = np.nonzero(FEMatricesA.MeshTri[n_nodes, :] == domain_tag)[0]
    tria_fluid = FEMatricesA.MeshTri[:, ind_tria_fluid].astype(int)  # triangles with nodes
    meshprops_b = FEMatricesA.MeshProps.b[:, ind_tria_fluid]  # coeff b of triangles
    meshprops_c = FEMatricesA.MeshProps.c[:, ind_tria_fluid]  # coeff c of triangles
    meshprops_delta = FEMatricesA.MeshProps.delta[ind_tria_fluid]  # delta of triangles

    num_tria_fluid = len(ind_tria_fluid)

    var_ind = tria_fluid[n_nodes - 1, :]
    xx_10 = FEMatricesA.MeshNodes[0, var_ind]
    yy_10 = FEMatricesA.MeshNodes[1, var_ind]

    # Coefficients to calculate dff/dx and dff/dy at 10 node (L_1=L_2=L_3=1/3)
    # by interpolation polynome ff(x,y)=sum_{i=1}^{10} ff_i*N_i
    cd_ndx = np.full(n_nodes, 1.5)
    cd_ndx[0:3] = -0.5
    cd_ndx[9] = 3.0
    cd_ndy = cd_ndx.copy()  # the same for y-differentiation (b_i->c_i)

    # work massive for Discrete Fourier transform
    exp_var_p = np.exp(1j * np.outer(theta_v, np.arange(ProcTE.NHarm)))
    exp_var_m = np.conj(exp_var_p)

    # ========================================================================
    # Main Calculations for given eigs
    # ========================================================================
    ifreq = ProcTE.ii_f
    ProcTE.freq = ProcTE.freq_count[ifreq]

    key_count_again = "y"
    while key_count_again == "y":
        # NOTE: the MATLAB script does not reset key_count_below inside the
        # while loop, which leads to an infinite loop after the first r_end
        # correction; here it is reset on every retry (intended behavior)
        key_count_below = "y"
        for iii_eig, ieig in enumerate(ProcTE.neigs):
            # Read Data of Pressure at all nodes (first half of the eigenvector)
            eivec_var = Results.REig_vecs[: Results.REig_vecs.shape[0] // 2, ieig]

            ivar_st, ivar_fn = ProcTE.stfn_layers[ii_l]  # 0-based half-open range
            eivec_fi = eivec_var[ivar_st:ivar_fn]

            eivec_fi_prev = None
            if ii_l > 0 and (
                CompStructA.Model.DomainType[ii_l - 1] == "fluid"
                and CompStructA.Model.DomainType[ii_l] == "fluid"
            ):
                ivar_st_prev, ivar_fn_prev = ProcTE.stfn_layers[ii_l - 1]
                eivec_fi_prev = eivec_var[ivar_st_prev:ivar_fn_prev]

            # ================================================================
            # Calculate ux, uy, uz, ur, uf, te at 10 nodes
            # ================================================================
            ux_10 = np.zeros(num_tria_fluid, dtype=complex)
            uy_10 = np.zeros(num_tria_fluid, dtype=complex)
            uz_10 = np.zeros(num_tria_fluid, dtype=complex)
            ur_10 = np.zeros(num_tria_fluid, dtype=complex)
            uf_10 = np.zeros(num_tria_fluid, dtype=complex)
            ff_10 = np.zeros(num_tria_fluid, dtype=complex)

            kk_10 = Results.REig_vals[ieig]
            d_nodes_comp = FEMatricesA.DNodesComp[ii_l]
            for in_10 in range(num_tria_fluid):
                # find potentials for all nodes (1-10) in elementary triangle
                et_nod = tria_fluid[0:n_nodes, in_10]
                ff = np.zeros(n_nodes, dtype=complex)
                for inn in range(n_nodes):
                    ii_m = np.nonzero(d_nodes_comp == et_nod[inn])[0]
                    if ii_m.size == 1:
                        ff[inn] = eivec_fi[ii_m[0]]
                    if len(CompStructA.Model.DomainType) == ii_l + 1 and ii_m.size == 0:
                        ff[inn] = 0.0
                    if eivec_fi_prev is not None:
                        ii_m1 = np.nonzero(FEMatricesA.DNodesComp[ii_l - 1] == et_nod[inn])[0]
                        if ii_m1.size == 1:
                            ff[inn] = eivec_fi_prev[ii_m1[0]]

                # ux displacement
                bb_2d = meshprops_b[:, in_10] / (2.0 * meshprops_delta[in_10])
                cbb = np.asarray(
                    [
                        bb_2d[0], bb_2d[1], bb_2d[2],
                        bb_2d[0], bb_2d[1], bb_2d[1], bb_2d[2], bb_2d[2], bb_2d[0],
                        bb_2d[0] + bb_2d[1] + bb_2d[2],
                    ]
                )
                ux_10[in_10] = -np.sum(cd_ndx * cbb * ff) / (-cone * Results.omega_val * CompStructA.Misc.F_conv)

                # uy displacement
                cc_2d = meshprops_c[:, in_10] / (2.0 * meshprops_delta[in_10])
                ccc = np.asarray(
                    [
                        cc_2d[0], cc_2d[1], cc_2d[2],
                        cc_2d[0], cc_2d[1], cc_2d[1], cc_2d[2], cc_2d[2], cc_2d[0],
                        cc_2d[0] + cc_2d[1] + cc_2d[2],
                    ]
                )
                uy_10[in_10] = -np.sum(cd_ndy * ccc * ff) / (-cone * Results.omega_val * CompStructA.Misc.F_conv)

                # uz displacement
                uz_10[in_10] = (kk_10 / (Results.omega_val * CompStructA.Misc.F_conv)) * ff[n_nodes - 1]

                # ur & uf displacement (quadrant logic is equivalent to atan2)
                fi_10 = np.arctan2(yy_10[in_10], xx_10[in_10])
                ur_10[in_10] = np.cos(fi_10) * ux_10[in_10] + np.sin(fi_10) * uy_10[in_10]
                uf_10[in_10] = -np.sin(fi_10) * ux_10[in_10] + np.cos(fi_10) * uy_10[in_10]

                # potential f
                ff_10[in_10] = ff[9]

            # ================================================================
            # ReCalculate ur_rf, uf_rf, uz_rf in Polar Grid
            # ================================================================
            ur_rf_10 = np.zeros((nrr, n_theta_v), dtype=complex)
            uf_rf_10 = np.zeros((nrr, n_theta_v), dtype=complex)
            uz_rf_10 = np.zeros((nrr, n_theta_v), dtype=complex)
            ff_rf_10 = np.zeros((nrr, n_theta_v), dtype=complex)

            # define interpolation functions in the rectangular grid
            fr_re_var = _scattered(xx_10, yy_10, np.real(ur_10))
            fr_im_var = _scattered(xx_10, yy_10, np.imag(ur_10))
            ff_re_var = _scattered(xx_10, yy_10, np.real(uf_10))
            ff_im_var = _scattered(xx_10, yy_10, np.imag(uf_10))
            fz_re_var = _scattered(xx_10, yy_10, np.real(uz_10))
            fz_im_var = _scattered(xx_10, yy_10, np.imag(uz_10))
            ff2_re_var = _scattered(xx_10, yy_10, np.real(ff_10))
            ff2_im_var = _scattered(xx_10, yy_10, np.imag(ff_10))

            for ir in range(nrr - 1, -1, -1):
                qxx, qyy = _pol2cart(theta_v, rr[ir])

                qzz_re = fr_re_var(qxx, qyy)
                qzz_im = fr_im_var(qxx, qyy)

                if ir == nrr - 1:
                    num_nan = np.isnan(qzz_re)  # check for nan answers
                    if np.count_nonzero(num_nan) >= 1:
                        # we have nan answer hence we'll correct r_max
                        rr[ir] = rr[ir] * 0.995
                        print("\tr_end_corr is corrected!")
                        key_count_below = "n"
                        break

                ur_rf_10[ir, :] = qzz_re + 1j * qzz_im
                qzz_re = ff_re_var(qxx, qyy)
                qzz_im = ff_im_var(qxx, qyy)
                uf_rf_10[ir, :] = qzz_re + 1j * qzz_im
                qzz_re = fz_re_var(qxx, qyy)
                qzz_im = fz_im_var(qxx, qyy)
                uz_rf_10[ir, :] = qzz_re + 1j * qzz_im
                qzz_re = ff2_re_var(qxx, qyy)
                qzz_im = ff2_im_var(qxx, qyy)
                ff_rf_10[ir, :] = qzz_re + 1j * qzz_im

            if key_count_below == "y":
                # ============================================================
                # Calculate T Energy
                # ============================================================
                var_c = 0.5 * rho_fluid * (2.0 * np.pi * ProcTE.freq * CompStructA.Misc.F_conv) ** 2
                var_df = dtheta_v / 2.0
                for ir in range(nrr):
                    var_te = (
                        np.abs(ur_rf_10[ir, 0 : n_theta_v - 1]) ** 2
                        + np.abs(uf_rf_10[ir, 0 : n_theta_v - 1]) ** 2
                        + np.abs(uz_rf_10[ir, 0 : n_theta_v - 1]) ** 2
                    )
                    ResFluid.TE_fer[iii_eig, ir] = var_c * np.sum(var_te) * var_df

                # ============================================================
                # Fourier decomposition of ur_rf, uf_rf, uz_rf by angle fi
                # for fixed r
                # ============================================================
                # I_i ~ 0.5*(f(x_{i-1})+f(x_i))*(x_i-x_{i-1})
                # Int f(x)=((f_0+f_n)/2+sum_{i=1}^{n-1} f_i)*dx, dx=(b-a)/n
                # we have that f_0=f_n !
                seg = slice(0, n_theta_v - 1)
                var_ur_pm = np.zeros((nrr, ProcTE.NHarm, 2), dtype=complex)
                var_uf_pm = np.zeros((nrr, ProcTE.NHarm, 2), dtype=complex)
                var_uz_pm = np.zeros((nrr, ProcTE.NHarm, 2), dtype=complex)
                for i_harm in range(ProcTE.NHarm):
                    for ir in range(nrr):
                        var_ur_pm[ir, i_harm, 0] = np.sum(ur_rf_10[ir, seg] * exp_var_p[seg, i_harm]) * var_df
                        var_uf_pm[ir, i_harm, 0] = np.sum(uf_rf_10[ir, seg] * exp_var_p[seg, i_harm]) * var_df
                        var_uz_pm[ir, i_harm, 0] = np.sum(uz_rf_10[ir, seg] * exp_var_p[seg, i_harm]) * var_df
                        var_ur_pm[ir, i_harm, 1] = np.sum(ur_rf_10[ir, seg] * exp_var_m[seg, i_harm]) * var_df
                        var_uf_pm[ir, i_harm, 1] = np.sum(uf_rf_10[ir, seg] * exp_var_m[seg, i_harm]) * var_df
                        var_uz_pm[ir, i_harm, 1] = np.sum(uz_rf_10[ir, seg] * exp_var_m[seg, i_harm]) * var_df

                ResFluid.cF_ur_rm_pm[iii_eig, :, :, :] = var_ur_pm / (2.0 * np.pi)
                ResFluid.cF_uf_rm_pm[iii_eig, :, :, :] = var_uf_pm / (2.0 * np.pi)
                ResFluid.cF_uz_rm_pm[iii_eig, :, :, :] = var_uz_pm / (2.0 * np.pi)

                if iii_eig == len(ProcTE.neigs) - 1:
                    key_count_again = "n"
            else:
                break  # retry all eigs with the corrected r_end

    ResFluid.rr = rr
    return ResFluid


# ============================================================================
# Solid layer processing, port of St61_Proc_Solid_TE_sp_SAFE.m
# ============================================================================
def St61_Proc_Solid_TE_sp_SAFE(CompStructA, FEMatricesA, ProcTE, Results):
    """Process kinetic energy of a solid (HTTI) layer for one frequency and all eigs."""
    ii_l = ProcTE.ilayer  # 0-based layer index (MATLAB ii_l is 1-based)
    n_nodes = int(CompStructA.Advanced.N_nodes)
    domain_tag = ii_l + 1  # domain tags in MeshTri stay 1-based

    # ========================================================================
    # Build Polar Grid
    # ========================================================================
    rr = [float(FEMatricesA.DomainRx[ii_l])]
    rr_min_prev_var = float(FEMatricesA.DomainRx[ii_l])

    if ii_l == 0:  # the smallest r of mesh
        rr_min_mesh = float(
            np.min(np.sqrt(FEMatricesA.MeshNodes[0, :] ** 2 + FEMatricesA.MeshNodes[1, :] ** 2))
        )
    else:
        rr_min_mesh = float(FEMatricesA.DomainRx[ii_l - 1])

    # find solid triangles and their nodes
    tri_solid_ind = FEMatricesA.MeshTri[n_nodes, :] == domain_tag
    tri_solid = FEMatricesA.MeshTri[0:n_nodes, tri_solid_ind].astype(int).copy()

    # set initial node with r=r_max and z=0
    r_find = float(FEMatricesA.DomainRx[ii_l])

    max_walk = 4 * tri_solid.shape[1] + 10  # safety guard against infinite loops
    i_count = 0
    while True:
        i_count += 1
        if i_count > max_walk:
            raise RuntimeError("rr-grid walk in the solid layer did not terminate")
        node_candidates = np.nonzero(FEMatricesA.MeshNodes[0, :] == r_find)[0]
        if node_candidates.size == 0:
            break
        node_find = int(node_candidates[0])

        num_tri_node_find = np.nonzero((tri_solid == node_find).any(axis=0))[0]
        if num_tri_node_find.size == 0:
            break
        tri_nodes = tri_solid[:, num_tri_node_find]

        rr_min_var = rr_min_prev_var
        tri_node_min = 0
        for iet in range(tri_nodes.shape[1]):
            tri_node_xx = FEMatricesA.MeshNodes[0, tri_nodes[:, iet]]
            tri_node_yy = FEMatricesA.MeshNodes[1, tri_nodes[:, iet]]
            rr_var3 = np.sqrt(tri_node_xx[0:3] ** 2 + tri_node_yy[0:3] ** 2)
            for ia in range(3):
                if rr_var3[ia] < rr_min_var:
                    rr_min_var = rr_var3[ia]
                    tri_node_min = iet

        tri_node_xx = FEMatricesA.MeshNodes[0, tri_nodes[:, tri_node_min]]
        tri_node_yy = FEMatricesA.MeshNodes[1, tri_nodes[:, tri_node_min]]
        rr_var = np.sqrt(tri_node_xx**2 + tri_node_yy**2)
        corner1, corner2, corner3 = _corner_ind_rr(rr_var)
        if tri_nodes[0, tri_node_min] == node_find:  # 1 node
            ind_rr_1, ind_rr_2, ind_rr = corner1
        elif tri_nodes[1, tri_node_min] == node_find:  # 2 node
            ind_rr_1, ind_rr_2, ind_rr = corner2
        else:  # 3 node
            ind_rr_1, ind_rr_2, ind_rr = corner3

        rr.extend([float(rr_var[ind_rr_1]), float(rr_var[ind_rr_2]), float(rr_var[ind_rr])])
        rr_min_prev_var = rr_var[ind_rr]

        if rr_var[ind_rr] <= rr_min_mesh * 1.01:
            break
        if len(rr) > 3 and rr_min_prev_var > rr[-3]:
            rr = rr[:-2]
            break

        r_find = float(tri_node_xx[ind_rr])
        tri_solid = np.delete(tri_solid, num_tri_node_find, axis=1)

    rr = np.asarray(rr)

    # tuning of rr(1) and rr(end)
    nodes_bound_ext = FEMatricesA.BNodesFull[ii_l]
    rr_bound_ext = np.min(
        FEMatricesA.MeshNodes[0, nodes_bound_ext] ** 2 + FEMatricesA.MeshNodes[1, nodes_bound_ext] ** 2
    )
    rr[0] = np.sqrt(rr_bound_ext) * 0.99

    if ii_l > 0:
        nodes_bound_int = FEMatricesA.BNodesFull[ii_l - 1]
        rr_bound_int = np.max(
            FEMatricesA.MeshNodes[0, nodes_bound_int] ** 2 + FEMatricesA.MeshNodes[1, nodes_bound_int] ** 2
        )
        rr[-1] = np.sqrt(rr_bound_int) * 1.01

    nrr = len(rr)
    # reverse rr grid
    rr = rr[::-1].copy()

    ResSolid = AttrDict()
    ResSolid.rr = rr

    # ========================================================================
    rho_solid = float(FEMatricesA.PhysProp[ii_l].rho) * 1000.0
    dtheta_v = np.pi / (int(CompStructA.Model.DomainNth[ii_l]) * ProcTE.DThetaInc)
    theta_v = np.arange(-np.pi, np.pi + 0.5 * dtheta_v, dtheta_v)
    n_theta_v = len(theta_v)
    # end of building of polar grid
    # ========================================================================

    # make memory for answer
    n_eigs = len(ProcTE.neigs)
    ResSolid.TE_fer = np.zeros((n_eigs, nrr))  # T Energy
    # last index: 0 - for plus, 1 - for minus
    ResSolid.cF_ur_rm_pm = np.zeros((n_eigs, nrr, ProcTE.NHarm, 2), dtype=complex)
    ResSolid.cF_uf_rm_pm = np.zeros((n_eigs, nrr, ProcTE.NHarm, 2), dtype=complex)
    ResSolid.cF_uz_rm_pm = np.zeros((n_eigs, nrr, ProcTE.NHarm, 2), dtype=complex)

    # Common parameters
    # Rectangular grid
    xx = FEMatricesA.MeshNodes[0, FEMatricesA.DNodes[ii_l]]
    yy = FEMatricesA.MeshNodes[1, FEMatricesA.DNodes[ii_l]]

    # work massive for Discrete Fourier transform
    exp_var_p = np.exp(1j * np.outer(theta_v, np.arange(ProcTE.NHarm)))
    exp_var_m = np.conj(exp_var_p)

    # ========================================================================
    # Main Calculations for given eigs
    # ========================================================================
    ifreq = ProcTE.ii_f
    ProcTE.freq = ProcTE.freq_count[ifreq]

    for iii_eig, ieig in enumerate(ProcTE.neigs):
        # ================================================================
        # Read Data of Solid at all nodes (first half of the eigenvector)
        # ================================================================
        eivec_var = Results.REig_vecs[: Results.REig_vecs.shape[0] // 2, ieig]

        ivar_st, ivar_fn = ProcTE.stfn_layers[ii_l]  # 0-based half-open range
        eivec_d = eivec_var[ivar_st:ivar_fn]
        eivec_d_prev = None
        if ii_l > 0 and (
            CompStructA.Model.DomainType[ii_l - 1] == "HTTI"
            and CompStructA.Model.DomainType[ii_l] == "HTTI"
        ):
            ivar_st_prev, ivar_fn_prev = ProcTE.stfn_layers[ii_l - 1]
            eivec_d_prev = eivec_var[ivar_st_prev:ivar_fn_prev]

        # ================================================================
        # Calculate ux, uy, uz, ur, uf, T Energy at all nodes
        # ================================================================
        n_d_nodes = len(FEMatricesA.DNodes[ii_l])
        ux = np.zeros(n_d_nodes, dtype=complex)
        uy = np.zeros(n_d_nodes, dtype=complex)
        uz = np.zeros(n_d_nodes, dtype=complex)
        ur = np.zeros(n_d_nodes, dtype=complex)
        uf = np.zeros(n_d_nodes, dtype=complex)

        # read ux, uy, uz displacements
        d_nodes_comp = FEMatricesA.DNodesComp[ii_l]
        d_nodes_rem = FEMatricesA.DNodesRem[ii_l]
        jj_n = 0
        for ii_n in range(n_d_nodes):
            ii_m = FEMatricesA.DNodes[ii_l][ii_n]

            if np.count_nonzero(d_nodes_comp == ii_m) == 1:
                ux[ii_n] = eivec_d[jj_n]
                uy[ii_n] = eivec_d[jj_n + 1]
                uz[ii_n] = eivec_d[jj_n + 2]
                jj_n += 3

            if np.count_nonzero(d_nodes_rem == ii_m) == 1:
                ux[ii_n] = 0.0
                uy[ii_n] = 0.0
                uz[ii_n] = 0.0

            if eivec_d_prev is not None:
                var_ind = np.nonzero(FEMatricesA.DNodesComp[ii_l - 1] == ii_m)[0]
                if var_ind.size == 1:
                    jj_nn = var_ind[0] * 3
                    ux[ii_n] = eivec_d_prev[jj_nn]
                    uy[ii_n] = eivec_d_prev[jj_nn + 1]
                    uz[ii_n] = eivec_d_prev[jj_nn + 2]

        # calculate ur, uf displacements (quadrant logic is equivalent to atan2)
        for ii_n in range(n_d_nodes):
            fi = np.arctan2(yy[ii_n], xx[ii_n])
            ur[ii_n] = np.cos(fi) * ux[ii_n] + np.sin(fi) * uy[ii_n]
            uf[ii_n] = -np.sin(fi) * ux[ii_n] + np.cos(fi) * uy[ii_n]

        # ================================================================
        # ReCalculate ur_rf, uf_rf, uz_rf in Polar Grid
        # ================================================================
        ur_rf = np.zeros((nrr, n_theta_v), dtype=complex)
        uf_rf = np.zeros((nrr, n_theta_v), dtype=complex)
        uz_rf = np.zeros((nrr, n_theta_v), dtype=complex)

        # define interpolation functions in the rectangular grid
        fr_re_var = _scattered(xx, yy, np.real(ur))
        fr_im_var = _scattered(xx, yy, np.imag(ur))
        ff_re_var = _scattered(xx, yy, np.real(uf))
        ff_im_var = _scattered(xx, yy, np.imag(uf))
        fz_re_var = _scattered(xx, yy, np.real(uz))
        fz_im_var = _scattered(xx, yy, np.imag(uz))

        for ir in range(nrr):
            qxx, qyy = _pol2cart(theta_v, rr[ir])
            qzz_re = fr_re_var(qxx, qyy)
            qzz_im = fr_im_var(qxx, qyy)
            ur_rf[ir, :] = qzz_re + 1j * qzz_im
            qzz_re = ff_re_var(qxx, qyy)
            qzz_im = ff_im_var(qxx, qyy)
            uf_rf[ir, :] = qzz_re + 1j * qzz_im
            qzz_re = fz_re_var(qxx, qyy)
            qzz_im = fz_im_var(qxx, qyy)
            uz_rf[ir, :] = qzz_re + 1j * qzz_im

        # ================================================================
        # Calculate T Energy
        # ================================================================
        var_c = 0.5 * rho_solid * (2.0 * np.pi * ProcTE.freq * CompStructA.Misc.F_conv) ** 2
        var_df = dtheta_v / 2.0
        for ir in range(nrr):
            var_te = (
                np.abs(ur_rf[ir, 0 : n_theta_v - 1]) ** 2
                + np.abs(uf_rf[ir, 0 : n_theta_v - 1]) ** 2
                + np.abs(uz_rf[ir, 0 : n_theta_v - 1]) ** 2
            )
            ResSolid.TE_fer[iii_eig, ir] = var_c * np.sum(var_te) * var_df

        # ================================================================
        # Fourier decomposition of ur_rf, uf_rf, uz_rf by angle fi for fixed r
        # ================================================================
        seg = slice(0, n_theta_v - 1)
        var_ur_pm = np.zeros((nrr, ProcTE.NHarm, 2), dtype=complex)
        var_uf_pm = np.zeros((nrr, ProcTE.NHarm, 2), dtype=complex)
        var_uz_pm = np.zeros((nrr, ProcTE.NHarm, 2), dtype=complex)
        for i_harm in range(ProcTE.NHarm):
            for ir in range(nrr):
                var_ur_pm[ir, i_harm, 0] = np.sum(ur_rf[ir, seg] * exp_var_p[seg, i_harm]) * var_df
                var_uf_pm[ir, i_harm, 0] = np.sum(uf_rf[ir, seg] * exp_var_p[seg, i_harm]) * var_df
                var_uz_pm[ir, i_harm, 0] = np.sum(uz_rf[ir, seg] * exp_var_p[seg, i_harm]) * var_df
                var_ur_pm[ir, i_harm, 1] = np.sum(ur_rf[ir, seg] * exp_var_m[seg, i_harm]) * var_df
                var_uf_pm[ir, i_harm, 1] = np.sum(uf_rf[ir, seg] * exp_var_m[seg, i_harm]) * var_df
                var_uz_pm[ir, i_harm, 1] = np.sum(uz_rf[ir, seg] * exp_var_m[seg, i_harm]) * var_df

        ResSolid.cF_ur_rm_pm[iii_eig, :, :, :] = var_ur_pm / (2.0 * np.pi)
        ResSolid.cF_uf_rm_pm[iii_eig, :, :, :] = var_uf_pm / (2.0 * np.pi)
        ResSolid.cF_uz_rm_pm[iii_eig, :, :, :] = var_uz_pm / (2.0 * np.pi)

    return ResSolid


# ============================================================================
# Driver, port of proc_aniso_TE.m
# ============================================================================
def proc_aniso_TE(dir_name, write_result="y", DThetaInc=3, NHarm=4, Polar_Grid="n"):
    """Process the raw results of gen_aniso into LayerTE and Slowness files.

    1. Calculate frequency dependencies of velocities and slownesses
    2. Build polar grid (r,f)
    3. On this polar grid the displacements ur and uf are calculated
    4. Calculate radial density of kinetic energy
    5. Calculate trigonometrical Fourier decomposition of ur,uf,uz
       (h=0,+-1, +-2, ..., +- NHarm-1)
    Results are saved in Slowness.npz and LayerTE-freq-layer.npz
    (freq in kHz, layer - its 1-based number).
    """
    dir_name = resolve_data_dir(dir_name)

    # Count for all layers, frequencies and eigenvalues
    ProcTE = AttrDict()
    ProcTE.write_result = write_result  # save (y) or not (n) results
    ProcTE.DThetaInc = DThetaInc  # decrease azimuthal space parameter DTheta
    ProcTE.NHarm = NHarm  # number of harmonics of the trigonometric Fourier transform
    ProcTE.Polar_Grid = Polar_Grid  # plot polar grid with mesh for each layer

    print("\n\n============================================================================")
    print("Proc_Aniso Program has been started!")
    t_start_prog = time.perf_counter()

    ProcTE.Dir_Name = str(dir_name)

    # Read common CompStruct data
    CompStructA = load_compstruct(dir_name)

    result_freq = np.zeros(int(CompStructA.Model.N_disp))
    ProcTE.freq_count = np.asarray(CompStructA.Model.f_array, dtype=float)
    ProcTE.nfreqs = len(ProcTE.freq_count)

    max_neigs = int(CompStructA.Advanced.num_eig_max)
    result_velocity = np.zeros((int(CompStructA.Model.N_disp), max_neigs))
    result_slowness = np.zeros((int(CompStructA.Model.N_disp), max_neigs))
    result_attenuation = np.zeros((int(CompStructA.Model.N_disp), max_neigs))

    ProcTE.max_neigs = max_neigs  # number of calculated eigs
    ProcTE.neigs = list(range(max_neigs))  # 0-based eigenvalue indices

    # Number of layers (0-based indices)
    ProcTE.nlayers = list(range(len(CompStructA.Model.DomainType)))

    # Give velocities in m/s
    slow_coef = 1.0
    if CompStructA.Config.SloUnits == "us/ft":
        slow_coef = CompStructA.Misc.S_conv / 1000.0  # 0.0254*12

    # ========================================================================
    for ii_f in range(ProcTE.nfreqs):  # frequency cycle
        t_start = time.perf_counter()
        print(
            f"Now fhe {ii_f + 1}-th frequency ({ProcTE.freq_count[ii_f]:.2f} kHz) "
            f"is processed out of {ProcTE.nfreqs}"
        )
        ProcTE.ii_f = ii_f

        # Read FEMatrices data (mesh, physical properties and etc.)
        freq = ProcTE.freq_count[ii_f]
        FEMatricesA = load_fematrices(dir_name / f"FEMatrices-{freq:g}.npz")

        # Loading the results for the specific point of eigenvalue variable
        Results = load_results(dir_name / f"Results-{freq:g}.npz")

        # The eigensolver may return fewer eigenvalues than requested;
        # process only the ones that are actually available
        ProcTE.neigs = list(range(min(max_neigs, Results.REig_vals.size)))

        # ================================================================
        # Calculate frequency dependencies of velocity and slowness
        var_mas = Results.REig_vals
        n_eig_avail = min(max_neigs, var_mas.size)
        result_freq[ii_f] = Results.omega_val / (2.0 * np.pi)
        result_velocity[ii_f, :n_eig_avail] = (
            CompStructA.Misc.F_conv * Results.omega_val / np.real(var_mas[:n_eig_avail])
        )
        result_slowness[ii_f, :n_eig_avail] = slow_coef * 10.0**6 / result_velocity[ii_f, :n_eig_avail]
        result_attenuation[ii_f, :n_eig_avail] = (
            -2.0 * np.imag(var_mas[:n_eig_avail]) / np.real(var_mas[:n_eig_avail])
        )

        # ================================================================
        # Prepare index partition of layers (0-based half-open ranges)
        ProcTE.stfn_layers = np.zeros((len(CompStructA.Model.DomainType), 2), dtype=int)
        var_beg = 0
        for ilayer in range(len(CompStructA.Model.DomainType)):
            if CompStructA.Model.DomainType[ilayer] == "fluid":
                ivar = len(FEMatricesA.DNodesComp[ilayer])  # potential
            elif CompStructA.Model.DomainType[ilayer] == "HTTI":
                ivar = 3 * len(FEMatricesA.DNodesComp[ilayer])  # displacement
            else:
                raise NotImplementedError(
                    f"Unsupported DomainType: {CompStructA.Model.DomainType[ilayer]}"
                )
            ProcTE.stfn_layers[ilayer] = (var_beg, var_beg + ivar)
            var_beg += ivar

        # ================================================================
        # Calculate trigonometrical Fourier decomposition of ur,uf,uz and
        # save in LayerTE-freq-layer.npz file
        for ilayer in ProcTE.nlayers:  # layer cycle
            ProcTE.ilayer = ilayer
            print(
                f"\tNow fhe {ilayer + 1}-th layer is processed out of "
                f"{len(ProcTE.nlayers)} (real {len(ProcTE.nlayers)}) layers"
            )

            # Fluid layer
            if CompStructA.Model.DomainType[ilayer] == "fluid":
                ResFluid = St61_Proc_Fluid_TE_sp_SAFE(CompStructA, FEMatricesA, ProcTE, Results)
                if ProcTE.write_result.lower() == "y":
                    fname = dir_name / f"LayerTE-{freq:g}-{ilayer + 1}.npz"
                    np.savez_compressed(
                        fname,
                        rr=ResFluid.rr,
                        TE_fer=ResFluid.TE_fer,
                        cF_ur_rm_pm=ResFluid.cF_ur_rm_pm,
                        cF_uf_rm_pm=ResFluid.cF_uf_rm_pm,
                        cF_uz_rm_pm=ResFluid.cF_uz_rm_pm,
                        NHarm=np.asarray([ProcTE.NHarm]),
                        neigs=np.asarray(ProcTE.neigs),
                        nlayers=np.asarray(ProcTE.nlayers),
                    )

            # Solid layer
            if CompStructA.Model.DomainType[ilayer] == "HTTI":
                ResSolid = St61_Proc_Solid_TE_sp_SAFE(CompStructA, FEMatricesA, ProcTE, Results)
                if ProcTE.write_result.lower() == "y":
                    fname = dir_name / f"LayerTE-{freq:g}-{ilayer + 1}.npz"
                    np.savez_compressed(
                        fname,
                        rr=ResSolid.rr,
                        TE_fer=ResSolid.TE_fer,
                        cF_ur_rm_pm=ResSolid.cF_ur_rm_pm,
                        cF_uf_rm_pm=ResSolid.cF_uf_rm_pm,
                        cF_uz_rm_pm=ResSolid.cF_uz_rm_pm,
                        NHarm=np.asarray([ProcTE.NHarm]),
                        neigs=np.asarray(ProcTE.neigs),
                        nlayers=np.asarray(ProcTE.nlayers),
                    )

        print(f"\tTime for this step = {time.perf_counter() - t_start:.1f}s")

    # ========================================================================
    # Save frequency dependencies of velocity and slowness in Slowness.npz file
    fname_slow = dir_name / "Slowness.npz"
    np.savez_compressed(
        fname_slow,
        Result_Freq=result_freq,
        Result_Velocity=result_velocity,
        Result_Slowness=result_slowness,
        Result_Attenuation=result_attenuation,
    )

    print(f"Time for Proc_Aniso TE = {time.perf_counter() - t_start_prog:.1f}s")
    print("============================================================================")
    return AttrDict(
        Result_Freq=result_freq,
        Result_Velocity=result_velocity,
        Result_Slowness=result_slowness,
        Result_Attenuation=result_attenuation,
    )
