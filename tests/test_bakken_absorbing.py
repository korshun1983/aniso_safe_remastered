"""Regression tests for Bakken frequency geometry and absorbing layers."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from aniso_safe.gen_aniso import _apply_additional_domain_logic
from aniso_safe.io_json import load_input_param
from aniso_safe.St1_SetModel import St1_SetModel
from aniso_safe.St2_PrepareModel_sp_SAFE import St2_PrepareModel_sp_SAFE
from aniso_safe.St3_1_PrepareBasicMatrices_sp_SAFE import (
    St3_PrepareBasicMatrices_sp_SAFE,
    _frequency_dependent_geometry,
)


def _bakken_input(add_type="abc", domain_nth=(12, 12)):
    root = Path(__file__).resolve().parents[1]
    user_model, _ = load_input_param(root / "models" / "bakken_b_00.json")
    user_model = deepcopy(user_model)
    user_model.Model.AddDomainType = add_type
    user_model.Model.DomainNth = list(domain_nth)
    user_model.Model.f_array = [1.0, 15.0]
    user_model.Advanced.VisualizeMesh = False
    user_model.Mesh.output = "no"
    input_param = St1_SetModel(user_model)
    return _apply_additional_domain_logic(input_param)


def _geometry_at(comp_struct, input_param, index):
    comp_struct.if_grid = index
    comp_struct.f_grid = np.asarray([1.0, 15.0], dtype=float)
    return _frequency_dependent_geometry(comp_struct, input_param)


def test_bakken_frequency_geometry_without_abc_matches_matlab():
    input_param = _bakken_input(add_type="none")
    comp = St2_PrepareModel_sp_SAFE(input_param)

    original_rx = list(input_param.Model.DomainRx)
    comp = _geometry_at(comp, input_param, 0)
    assert comp.Model.DomainRx == pytest.approx([0.1, 4.439824963922944])
    comp = _geometry_at(comp, input_param, 1)
    assert comp.Model.DomainRx == pytest.approx([0.1, 0.3893216642615296])

    # MATLAB InputParam remains the reference value object throughout the loop.
    assert input_param.Model.DomainRx == original_rx == [0.1, 2.0]


def test_bakken_frequency_geometry_with_external_abc_matches_matlab():
    input_param = _bakken_input(add_type="abc")
    comp = St2_PrepareModel_sp_SAFE(input_param)

    comp = _geometry_at(comp, input_param, 0)
    assert comp.Model.DomainRx == pytest.approx(
        [0.1, 4.439824963922944, 6.6097374458844165]
    )
    assert comp.Model.AddDomainL_m == pytest.approx(2.169912481961472)

    comp = _geometry_at(comp, input_param, 1)
    assert comp.Model.DomainRx == pytest.approx(
        [0.1, 0.3893216642615296, 0.5339824963922943]
    )
    assert comp.Model.AddDomainL_m == pytest.approx(0.1446608321307648)
    assert input_param.Model.DomainRx == [0.1, 2.0]


@pytest.mark.parametrize("add_type", ["abc", "pml", "pml+abc"])
def test_absorbing_layer_st3_preserves_complex_matrices(tmp_path, add_type):
    # A small angular mesh is enough to exercise the complete element and sparse
    # assembly paths without making the regression suite expensive.
    input_param = _bakken_input(add_type=add_type, domain_nth=(4, 4))
    input_param.Model.f_array = np.asarray([1.0])
    input_param.Model.N_disp = 1
    comp = St2_PrepareModel_sp_SAFE(input_param)
    comp.f_grid = np.asarray([1.0])
    comp.if_grid = 0

    comp, _, _, full = St3_PrepareBasicMatrices_sp_SAFE(comp, input_param, tmp_path)

    assert np.iscomplexobj(full.K1Matrix.data)
    assert np.iscomplexobj(full.K2Matrix.data)
    assert np.iscomplexobj(full.K3Matrix.data)
    assert np.max(np.abs(full.K1Matrix.data.imag)) > 0.0
    if add_type in ("pml", "pml+abc"):
        assert np.iscomplexobj(full.MMatrix.data)
        assert np.max(np.abs(full.MMatrix.data.imag)) > 0.0


def test_matlab_results_loader_reads_reig_vals_not_vectors(tmp_path):
    from scipy.io import savemat
    from aniso_safe.postproc import load_result_file

    path = tmp_path / "Results-1.mat"
    expected = np.asarray([1.25 + 0.1j, 2.5 - 0.2j])
    savemat(
        path,
        {
            "Results": {
                # Put vectors first deliberately: the loader must prefer vals.
                "REig_vecs": np.arange(16, dtype=float).reshape(4, 4),
                "REig_vals": np.diag(expected),
            }
        },
    )
    record = load_result_file(path)
    assert record.eigenvalues == pytest.approx(expected)
