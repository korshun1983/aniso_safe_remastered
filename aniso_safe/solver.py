"""Main orchestration ported from gen_aniso.m."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .io_json import load_input_param
from .mesh import PrepareMesh_sp_SAFE
from .st1 import St1_SetModel
from .st2 import St2_PrepareModel_sp_SAFE
from .structures import AttrDict, to_plain


def _apply_additional_domain_logic(InputParam):
    """Apply AddDomain extension logic from gen_aniso.m."""
    model = InputParam.Model
    model.AddDomain_Exist = "none"
    model.AddDomainType = str(model.get("AddDomainType", "none")).lower()
    if model.AddDomainType == "abc+pml":
        model.AddDomainType = "pml+abc"
    if model.AddDomainType != "none":
        if model.AddDomainType in ("pml", "abc", "pml+abc", "same"):
            model.AddDomain_Exist = "yes"
        else:
            raise ValueError("AddDomainType is not set")

    if model.AddDomain_Exist == "yes" and str(model.get("AddDomainLoc", "ext")).lower() == "ext":
        model.DomainTheta.append(model.DomainTheta[-1])
        model.DomainEcc.append(model.DomainEcc[-1])
        model.DomainEccAngle.append(model.DomainEccAngle[-1])
        model.DomainParam.append(list(model.DomainParam[-1]))
        model.DomainType.append(model.DomainType[-1])
        model.BCType[-1] = "SSstiff"
        model.BCType.append("rigid")
        model.DomainNth.append(model.DomainNth[-1])
        outer_rx = float(model.DomainRx[-1])
        outer_ry = float(model.DomainRy[-1])
        add_l = float(model.get("AddDomainL", 0.0))
        if str(model.get("LDomain_in_LSH", "yes")).lower() == "yes":
            # The wavelength-based extension requires the SH asymptote and is
            # finalized in St2 after CheckAsymptote. Keep the relative length
            # here and use it as an additive radius increment at this stage.
            model.DomainRx.append(outer_rx + add_l)
            model.DomainRy.append(outer_ry + add_l)
        else:
            model.DomainRx.append(outer_rx + add_l)
            model.DomainRy.append(outer_ry + add_l)
    return InputParam


def St3_PrepareBasicMatrices_sp_SAFE(CompStruct, InputParam, output_dir):
    """Prepare mesh data for the current frequency.

    The heavy SAFE matrix kernels are intentionally isolated behind this
    function boundary so the remaining numerical port can replace the body
    without changing the orchestration contract.
    """
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
    BasicMatrices = AttrDict(status="pending numerical kernels")
    FullMatrices = AttrDict(status="pending numerical kernels")
    return CompStruct, BasicMatrices, FEMatrices, FullMatrices


def St4_ComputeSolution_sp_SAFE(CompStruct, BasicMatrices, FEMatrices, FullMatrices):
    """Compute SAFE spectrum placeholder with the MATLAB function contract.

    Returning an explicit empty result keeps the pipeline testable while the
    generalized eigenvalue kernels are ported; it does not fabricate modes.
    """
    return AttrDict(
        freq_khz=float(CompStruct.f_grid[CompStruct.if_grid]),
        eigenvalues=np.zeros((0,), dtype=complex),
        eigenvectors=np.zeros((0, 0), dtype=complex),
        status="pending numerical kernels",
    )


def gen_aniso(model_json, output_dir=None, keep_output=True, mesh_output=None):
    """Run the main computation workflow, port of gen_aniso.m."""
    print("\n\n============================================================================")
    print("Gen_Aniso Program has been started!")
    user_model, model_path = load_input_param(model_json)
    if mesh_output is not None:
        if "Mesh" not in user_model:
            user_model.Mesh = AttrDict()
        user_model.Mesh.output = str(mesh_output).lower()
    print(f"\tThe used model file is {model_path}")
    t_start_prog = time.perf_counter()

    if output_dir is None:
        output_dir = model_path.with_suffix("")
    output_dir = Path(output_dir)
    work_dir = output_dir / "output"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    InputParam = St1_SetModel(user_model)
    InputParam = _apply_additional_domain_logic(InputParam)
    CompStruct = St2_PrepareModel_sp_SAFE(InputParam)
    CompStruct.f_grid = CompStruct.Model.f_array

    saved_files = []
    for if_grid in range(int(CompStruct.Model.N_disp)):
        CompStruct.if_grid = if_grid
        print(f"\tNow computing the {if_grid + 1}th point on dispersion curve out of {CompStruct.Model.N_disp}")
        t_start_st3 = time.perf_counter()
        CompStruct, BasicMatrices, FEMatrices, FullMatrices = St3_PrepareBasicMatrices_sp_SAFE(
            CompStruct, InputParam, work_dir
        )
        print(f"\t\tTime for St3 Program = {time.perf_counter() - t_start_st3:.1f}s")

        freq = CompStruct.f_grid[if_grid]
        fem_file = work_dir / f"FEMatrices-{freq:g}.npz"
        np.savez_compressed(
            fem_file,
            MeshNodes=FEMatrices.MeshNodes,
            BoundaryEdges=FEMatrices.BoundaryEdges,
            MeshTri=FEMatrices.MeshTri,
            delta=FEMatrices.MeshProps.delta,
            a=FEMatrices.MeshProps.a,
            b=FEMatrices.MeshProps.b,
            c=FEMatrices.MeshProps.c,
            DomainRx=FEMatrices.DomainRx,
            DomainRy=FEMatrices.DomainRy,
        )
        saved_files.append(fem_file)

        t_start_st4 = time.perf_counter()
        Results = St4_ComputeSolution_sp_SAFE(CompStruct, BasicMatrices, FEMatrices, FullMatrices)
        res_file = work_dir / f"Results-{freq:g}.npz"
        np.savez_compressed(
            res_file,
            eigenvalues=Results.eigenvalues,
            eigenvectors=Results.eigenvectors,
            status=np.asarray([Results.status]),
        )
        saved_files.append(res_file)
        print(f"\t\tTime for St4 Program = {time.perf_counter() - t_start_st4:.1f}s")

    comp_file = work_dir / "CompStruct.npz"
    np.savez_compressed(comp_file, CompStruct=np.asarray([str(to_plain(CompStruct))], dtype=object))
    saved_files.append(comp_file)

    if keep_output:
        final_dir = output_dir / "results"
        final_dir.mkdir(parents=True, exist_ok=True)
        for src in work_dir.iterdir():
            shutil.copy2(src, final_dir / src.name)

    print(f"Time for Gen_Aniso Program = {time.perf_counter() - t_start_prog:.1f}s")
    print("============================================================================")
    return CompStruct, saved_files
