"""Main orchestration ported from gen_aniso.m."""

from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from .io_json import load_input_param
from .St1_SetModel import St1_SetModel
from .St2_PrepareModel_sp_SAFE import St2_PrepareModel_sp_SAFE
from .St3_1_PrepareBasicMatrices_sp_SAFE import St3_PrepareBasicMatrices_sp_SAFE
from .St4_ComputeSolution_sp_SAFE import St4_ComputeSolution_sp_SAFE
from .structures import AttrDict, to_plain


def _save_npz(path, compressed=True, **arrays):
    """Save an npz file with a couple of retries.

    The result files are the essential output of the whole run, so a
    transient filesystem hiccup is retried before giving up.
    """
    saver = np.savez_compressed if compressed else np.savez
    last_err = None
    for _ in range(3):
        try:
            saver(path, **arrays)
            if Path(path).exists():
                return
        except OSError as err:
            last_err = err
        time.sleep(0.5)
    if last_err is not None:
        raise last_err
    raise OSError(f"Could not save {path}")


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
        # Deliberately do NOT append DomainRx/DomainRy here.  The MATLAB
        # gen_aniso.m only appends the additional domain metadata; its radii
        # are created in St3 for every frequency.  This matters both for
        # LDomain_in_LSH='yes' and for a fixed-thickness PML/ABC layer.
    return InputParam
def gen_aniso(model_json, output_dir=None, mesh_output=None):
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
    # All artifacts are written directly into the requested output directory.
    # A previous run is wiped only when the directory is clearly a former
    # output directory (marker: CompStruct.npz), never an arbitrary folder.
    work_dir = output_dir
    if work_dir.exists() and (work_dir / "CompStruct.npz").exists():
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
        n_domain = int(CompStruct.Data.N_domain)
        fem_save = dict(
            MeshNodes=FEMatrices.MeshNodes,
            BoundaryEdges=FEMatrices.BoundaryEdges,
            MeshTri=FEMatrices.MeshTri,
            delta=FEMatrices.MeshProps.delta,
            a=FEMatrices.MeshProps.a,
            b=FEMatrices.MeshProps.b,
            c=FEMatrices.MeshProps.c,
            DomainRx=FEMatrices.DomainRx,
            DomainRy=FEMatrices.DomainRy,
            N_domain=np.asarray([n_domain]),
        )
        # Per-domain data needed by the TE postprocessing (St61_* scripts)
        for ii_d in range(n_domain):
            fem_save[f"DElements_{ii_d}"] = FEMatrices.DElements[ii_d]
            fem_save[f"DEMeshProps_delta_{ii_d}"] = FEMatrices.DEMeshProps[ii_d].delta
            fem_save[f"DEMeshProps_a_{ii_d}"] = FEMatrices.DEMeshProps[ii_d].a
            fem_save[f"DEMeshProps_b_{ii_d}"] = FEMatrices.DEMeshProps[ii_d].b
            fem_save[f"DEMeshProps_c_{ii_d}"] = FEMatrices.DEMeshProps[ii_d].c
            fem_save[f"DNodes_{ii_d}"] = FEMatrices.DNodes[ii_d]
            fem_save[f"DNodesRem_{ii_d}"] = FEMatrices.DNodesRem[ii_d]
            fem_save[f"DNodesComp_{ii_d}"] = FEMatrices.DNodesComp[ii_d]
            fem_save[f"DTakeFromVarPos_{ii_d}"] = FEMatrices.DTakeFromVarPos[ii_d]
            fem_save[f"DPutToVarPos_{ii_d}"] = FEMatrices.DPutToVarPos[ii_d]
            fem_save[f"DZeroVarPos_{ii_d}"] = FEMatrices.DZeroVarPos[ii_d]
            # PhysProp structures are stored as pickled plain dicts
            fem_save[f"PhysProp_{ii_d}"] = np.frombuffer(
                pickle.dumps(to_plain(FEMatrices.PhysProp[ii_d])), dtype=np.uint8
            )
        for ii_int, b_nodes in enumerate(FEMatrices.BNodes):
            fem_save[f"BNodes_{ii_int}"] = b_nodes
        for ii_int, b_nodes_full in enumerate(FEMatrices.BNodesFull):
            if b_nodes_full is not None:
                fem_save[f"BNodesFull_{ii_int}"] = b_nodes_full
        _save_npz(fem_file, **fem_save)
        saved_files.append(fem_file)

        t_start_st4 = time.perf_counter()
        Results = St4_ComputeSolution_sp_SAFE(CompStruct, BasicMatrices, FEMatrices, FullMatrices)
        res_file = work_dir / f"Results-{freq:g}.npz"
        _save_npz(
            res_file,
            eigenvalues=Results.eigenvalues,
            eigenvectors=Results.eigenvectors,
            omega_val=np.asarray([Results.omega_val]),
            status=np.asarray([Results.status]),
        )
        saved_files.append(res_file)
        if Results.eigenvalues.size == 0:
            print(f"\t\tSpectrum is empty ({Results.status}).")
        print(f"\t\tTime for St4 Program = {time.perf_counter() - t_start_st4:.1f}s")

    comp_file = work_dir / "CompStruct.npz"
    # Store the whole CompStruct as a pickled plain dict so that the
    # postprocessing scripts can restore it without rerunning St1-St3
    _save_npz(
        comp_file,
        CompStruct=np.frombuffer(pickle.dumps(to_plain(CompStruct)), dtype=np.uint8),
    )
    saved_files.append(comp_file)

    print(f"Time for Gen_Aniso Program = {time.perf_counter() - t_start_prog:.1f}s")
    print("============================================================================")
    return CompStruct, saved_files
