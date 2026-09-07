"""Smoke tests for the runnable port scaffold."""

from pathlib import Path

import numpy as np

from aniso_safe.io_json import load_input_param
from aniso_safe.mesh import PrepareMesh_sp_SAFE
from aniso_safe.st1 import St1_SetModel
from aniso_safe.st2 import St2_PrepareModel_sp_SAFE


def test_json_mesh_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    user_model, _ = load_input_param(root / "models" / "bakken_b_00.json")
    input_param = St1_SetModel(user_model)
    comp_struct = St2_PrepareModel_sp_SAFE(input_param)
    comp_struct.if_grid = 0
    comp_struct.f_grid = comp_struct.Model.f_array
    mesh_nodes, boundary_edges, mesh_tri, mesh_props, comp_struct = PrepareMesh_sp_SAFE(
        comp_struct, out_png=tmp_path / "mesh.png"
    )
    assert mesh_nodes.shape[0] == 2
    assert mesh_tri.shape[0] == 11
    assert mesh_props.delta.size == mesh_tri.shape[1]
    assert np.all(mesh_props.delta > 0)
    assert boundary_edges.shape[0] == 3
    assert (tmp_path / "mesh.png").exists()
