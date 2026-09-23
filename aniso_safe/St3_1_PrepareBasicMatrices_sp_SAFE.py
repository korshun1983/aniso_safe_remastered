"""Step 3: FE matrix preparation, port of St3_1_PrepareBasicMatrices_sp_SAFE.m."""

from __future__ import annotations

import numpy as np

from . import fematrices as fe
from .basic import AssembleBasicMatrices_sp_SAFE, convolution_cache_dir
from .mesh import PrepareMesh_sp_SAFE
from .structures import AttrDict


def _set_or_append(values, index, value):
    """MATLAB-like indexed assignment that can extend a 1-D vector."""
    if index < len(values):
        values[index] = float(value)
    elif index == len(values):
        values.append(float(value))
    else:
        raise IndexError(f"Cannot assign geometry index {index} to vector of length {len(values)}")


def _frequency_dependent_geometry(CompStruct, InputParam):
    """Reproduce the geometry update at lines 71-113 of MATLAB St3.

    DomainRx/DomainRy in the input file are not always literal radii.  With
    LDomain_in_LSH='yes', the last supplied value is a number of SH
    wavelengths.  For an external PML/ABC domain MATLAB also creates the
    additional outer radius here, inside the frequency loop.
    """
    model = CompStruct.Model
    ref_model = InputParam.Model
    n_domain = int(CompStruct.Data.N_domain)
    l_mode = str(model.get("LDomain_in_LSH", "none")).lower()
    if l_mode == "no":
        l_mode = "none"
    add_exists = str(model.get("AddDomain_Exist", "none")).lower() == "yes"
    add_loc = str(model.get("AddDomainLoc", "ext")).lower()

    # Start every frequency from the untouched input geometry.  This is the
    # Python equivalent of MATLAB copy-on-write semantics for InputParam.
    rx = [float(v) for v in ref_model.DomainRx]
    ry = [float(v) for v in ref_model.DomainRy]

    if l_mode == "yes":
        if "Asymp" not in CompStruct or "V_SH" not in CompStruct.Asymp:
            raise ValueError("LDomain_in_LSH='yes' requires the V_SH asymptote")
        freq = float(CompStruct.f_grid[CompStruct.if_grid])
        freq_hz = freq * float(CompStruct.Misc.F_conv)
        if freq_hz <= 0.0:
            raise ValueError("Frequency must be positive for LDomain_in_LSH='yes'")
        var_wl = float(CompStruct.Asymp.V_SH) * 1.0e3 / freq_hz

        if add_exists:
            model.AddDomainL_m = float(model.AddDomainL) * var_wl
            if add_loc == "ext":
                # MATLAB (1-based):
                # DomainR(nl-1)=InputR(nl-2)+InputR(nl-1)*lambda_SH
                # DomainR(nl)=DomainR(nl-1)+AddDomainL*lambda_SH
                core_rx = float(ref_model.DomainRx[n_domain - 3]) + float(ref_model.DomainRx[n_domain - 2]) * var_wl
                core_ry = float(ref_model.DomainRy[n_domain - 3]) + float(ref_model.DomainRy[n_domain - 2]) * var_wl
                _set_or_append(rx, n_domain - 2, core_rx)
                _set_or_append(ry, n_domain - 2, core_ry)
                _set_or_append(rx, n_domain - 1, core_rx + float(model.AddDomainL_m))
                _set_or_append(ry, n_domain - 1, core_ry + float(model.AddDomainL_m))
            elif add_loc == "int":
                # No extra material domain is appended for an internal PML;
                # the last physical domain contains the absorbing sublayer.
                outer_rx = float(ref_model.DomainRx[n_domain - 2]) + float(ref_model.DomainRx[n_domain - 1]) * var_wl
                outer_ry = float(ref_model.DomainRy[n_domain - 2]) + float(ref_model.DomainRy[n_domain - 1]) * var_wl
                _set_or_append(rx, n_domain - 1, outer_rx)
                _set_or_append(ry, n_domain - 1, outer_ry)
            else:
                raise ValueError(f"Unsupported AddDomainLoc: {add_loc}")
        else:
            # Last domain extent is specified as a number of SH wavelengths
            # measured from the previous interface.
            outer_rx = float(ref_model.DomainRx[n_domain - 2]) + float(ref_model.DomainRx[n_domain - 1]) * var_wl
            outer_ry = float(ref_model.DomainRy[n_domain - 2]) + float(ref_model.DomainRy[n_domain - 1]) * var_wl
            _set_or_append(rx, n_domain - 1, outer_rx)
            _set_or_append(ry, n_domain - 1, outer_ry)

    elif l_mode in ("none", "fixed"):
        if add_exists:
            model.AddDomainL_m = float(model.AddDomainL)
            if add_loc == "ext":
                _set_or_append(rx, n_domain - 1, rx[n_domain - 2] + float(model.AddDomainL_m))
                _set_or_append(ry, n_domain - 1, ry[n_domain - 2] + float(model.AddDomainL_m))
            elif add_loc == "int":
                varx = rx[n_domain - 1] - float(model.AddDomainL_m)
                vary = ry[n_domain - 1] - float(model.AddDomainL_m)
                if varx <= rx[n_domain - 2] or vary <= ry[n_domain - 2]:
                    raise ValueError("Error in specifying model geometry: internal absorbing layer overlaps previous domain")
            else:
                raise ValueError(f"Unsupported AddDomainLoc: {add_loc}")
    else:
        raise ValueError(f"Unsupported LDomain_in_LSH value: {model.LDomain_in_LSH}")

    if len(rx) != n_domain or len(ry) != n_domain:
        raise ValueError(
            f"Geometry/domain mismatch after frequency update: N_domain={n_domain}, "
            f"len(DomainRx)={len(rx)}, len(DomainRy)={len(ry)}"
        )
    model.DomainRx = rx
    model.DomainRy = ry
    return CompStruct


def St3_PrepareBasicMatrices_sp_SAFE(CompStruct, InputParam, output_dir):
    """Prepare all FE matrices for the current frequency.

    Port of St3_1_PrepareBasicMatrices_sp_SAFE.m: basic matrices, mesh,
    per-domain matrix blocks, interface condition matrices, full matrix
    assembly and removal of the redundant variables.
    """
    # Introduce basic matrices, like L_x, L_y, L_z
    # (the convolution integrals are cached in the system temp directory,
    # they are identical for every frequency and every model with the
    # same element orders)
    BasicMatrices = AssembleBasicMatrices_sp_SAFE(CompStruct, cache_dir=convolution_cache_dir())
    BasicMatrices.f_grid = CompStruct.f_grid

    # MATLAB updates the radial geometry inside the frequency loop, before
    # meshing.  This applies both with and without an additional PML/ABC
    # domain and is essential for Bakken-B.
    CompStruct = _frequency_dependent_geometry(CompStruct, InputParam)

    # Assembling the mesh
    freq = CompStruct.f_grid[CompStruct.if_grid]
    mesh_png = output_dir / f"mesh-{freq:g}.png"
    mesh_nodes, boundary_edges, mesh_tri, mesh_props, CompStruct = PrepareMesh_sp_SAFE(
        CompStruct, out_png=mesh_png
    )
    FEMatrices = AttrDict(
        MeshNodes=mesh_nodes,
        BoundaryEdges=boundary_edges,
        MeshTri=mesh_tri,
        MeshProps=mesh_props,
        DomainRx=np.asarray(CompStruct.Model.DomainRx, dtype=float),
        DomainRy=np.asarray(CompStruct.Model.DomainRy, dtype=float),
    )

    n_domain = int(CompStruct.Data.N_domain)
    FEMatrices.PhysProp = [None] * n_domain
    FEMatrices.DElements = [None] * n_domain
    FEMatrices.DEMeshProps = [None] * n_domain
    FEMatrices.DNodes = [None] * n_domain
    FEMatrices.DNodesRem = [np.zeros(0, dtype=int)] * n_domain
    FEMatrices.DNodesComp = [np.zeros(0, dtype=int)] * n_domain
    FEMatrices.DTakeFromVarPos = [np.zeros(0, dtype=int)] * n_domain
    FEMatrices.DPutToVarPos = [np.zeros(0, dtype=int)] * n_domain
    FEMatrices.DZeroVarPos = [np.zeros(0, dtype=int)] * n_domain
    FEMatrices.K1Matrix_d = [None] * n_domain
    FEMatrices.K2Matrix_d = [None] * n_domain
    FEMatrices.K3Matrix_d = [None] * n_domain
    FEMatrices.MMatrix_d = [None] * n_domain
    FEMatrices.PMatrix_d = [None] * n_domain
    FEMatrices.B1tCB2Matrix_d = [None] * n_domain
    FEMatrices.B2tCB1Matrix_d = [None] * n_domain
    FEMatrices.BNodes = []
    FEMatrices.BNodesFull = [None] * n_domain
    FEMatrices.PMatrixD12 = [None] * (n_domain - 1)
    FEMatrices.PMatrixD21 = [None] * (n_domain - 1)

    # Preparing the matrices, which are necessary for the description of
    # each domain (mesh domain tags are one-based, as in MATLAB)
    domain_tags = mesh_tri[10, :]
    for ii_d in range(n_domain):
        # preparing the physical properties of the domain
        FEMatrices.PhysProp[ii_d] = fe.call_method(
            CompStruct.Methods.PreparePhysProp[ii_d], CompStruct, ii_d
        )

        # finding the elements, which belong to the domain
        domain_mask = domain_tags == ii_d + 1
        FEMatrices.DElements[ii_d] = mesh_tri[:, domain_mask]
        # finding their properties
        FEMatrices.DEMeshProps[ii_d] = AttrDict(
            delta=mesh_props.delta[domain_mask],
            a=mesh_props.a[:, domain_mask],
            b=mesh_props.b[:, domain_mask],
            c=mesh_props.c[:, domain_mask],
        )
        # finding the nodes, which belong to the domain
        FEMatrices.DNodes[ii_d] = np.unique(FEMatrices.DElements[ii_d][0:10, :])

        # computing the blocks of the mass and stiffness matrices for each domain
        FEMatrices = fe.call_method(
            CompStruct.Methods.MatricesParts_sp_SAFE[ii_d],
            CompStruct, BasicMatrices, FEMatrices, ii_d,
        )

    # Preparing the matrices, which are necessary for the description of
    # each boundary (interface condition matrices)
    for ii_int in range(n_domain - 1):
        # finding the nodes, which belong to the boundary
        b_nodes_el = boundary_edges[0:2, boundary_edges[2, :] == ii_int + 1]
        FEMatrices.BNodes.append(np.unique(b_nodes_el))
        FEMatrices = fe.call_method(
            CompStruct.Methods.IC_Matrices_sp_SAFE[ii_int],
            CompStruct, BasicMatrices, FEMatrices, ii_int, ii_int, ii_int + 1,
        )

    # Calculating indices, which describe start positions of various blocks
    BasicMatrices = fe.FindPos_sp_SAFE(CompStruct, FEMatrices, BasicMatrices)

    # Construct the full matrices - insert the blocks for the first domain
    FullMatrices = AttrDict(
        K1Matrix=FEMatrices.K1Matrix_d[0].copy(),
        K2Matrix=FEMatrices.K2Matrix_d[0].copy(),
        K3Matrix=FEMatrices.K3Matrix_d[0].copy(),
        MMatrix=FEMatrices.MMatrix_d[0].copy(),
        PMatrix=FEMatrices.PMatrix_d[0].copy(),
    )
    FEMatrices.DNodesRem[0] = np.zeros(0, dtype=int)
    FEMatrices.DNodesComp[0] = FEMatrices.DNodes[0]

    # Assembling the blocks of the matrices together and inserting the
    # boundary conditions at the domain interfaces
    for ii_int in range(n_domain - 1):
        FEMatrices, FullMatrices = fe.call_method(
            CompStruct.Methods.AssembleFullMatrices[ii_int],
            CompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_int, ii_int + 1,
        )

    # Assembling the blocks of the matrices together and inserting the
    # boundary conditions at the outer interface
    ii_int = n_domain - 1
    b_nodes_el = boundary_edges[0:2, boundary_edges[2, :] == n_domain]
    FEMatrices.BNodes.append(np.unique(b_nodes_el))
    FEMatrices, FullMatrices = fe.call_method(
        CompStruct.Methods.AssembleFullMatrices[ii_int],
        CompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_int, ii_int + 1,
    )

    # Remove the rows and columns of the full matrices corresponding to the
    # nodes, which do not participate in the computation
    FEMatrices, FullMatrices = fe.RemoveRedundantVariables_SAFE(
        CompStruct, BasicMatrices, FEMatrices, FullMatrices
    )
    return CompStruct, BasicMatrices, FEMatrices, FullMatrices
