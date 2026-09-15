"""Step 3: FE matrix preparation, port of St3_1_PrepareBasicMatrices_sp_SAFE.m."""

from __future__ import annotations

import numpy as np

from . import fematrices as fe
from .basic import AssembleBasicMatrices_sp_SAFE, convolution_cache_dir
from .mesh import PrepareMesh_sp_SAFE
from .structures import AttrDict

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

    # NOTE: the frequency-dependent adjustment of the additional domain
    # geometry (LDomain_in_LSH == 'yes') from the MATLAB St3 is used only
    # together with the PML/ABC additional domains and is ported together
    # with the KM_el_matrix_HTTI_PML/ABC kernels.

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
