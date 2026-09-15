"""JSON model loading for the Python port of ANISO_SAFE."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .structures import as_attrdict


def _parse_matlab_range(value):
    """Parse MATLAB-like ranges such as '0.5:0.25:15' or '0.5 1:5'."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    tokens = text.replace(",", " ").split()
    numbers = []
    range_re = re.compile(r"^(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?)$")
    two_re = re.compile(r"^(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?)$")
    for token in tokens:
        m3 = range_re.match(token)
        if m3:
            start, step, stop = map(float, m3.groups())
            count = int(np.floor((stop - start) / step + 1e-12)) + 1
            numbers.extend((start + i * step for i in range(count)))
            continue
        m2 = two_re.match(token)
        if m2:
            start, stop = map(float, m2.groups())
            step = 1.0 if stop >= start else -1.0
            count = int(np.floor((stop - start) / step + 1e-12)) + 1
            numbers.extend((start + i * step for i in range(count)))
            continue
        numbers.append(float(token))
    return numbers


def _parse_start_step_stop(value):
    """Parse a frequency object {'start': a, 'step': b, 'stop': c}."""
    if not isinstance(value, dict):
        return value
    start = float(value["start"])
    step = float(value.get("step", 1.0))
    stop = float(value["stop"])
    if step == 0:
        raise ValueError("f_array step must be nonzero")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    if count < 1:
        raise ValueError("f_array start/step/stop produce an empty range")
    return [start + i * step for i in range(count)]


def _normalize_arrays(model):
    """Convert numeric model fields to lists/arrays with MATLAB semantics."""
    for field in ("DomainRx", "DomainRy", "DomainTheta", "DomainEcc", "DomainEccAngle", "DomainNth"):
        if field in model:
            model[field] = [float(v) for v in np.atleast_1d(model[field])]
    if "f_array" not in model and {"f_start", "f_stop"} <= set(model):
        model["f_array"] = {
            "start": model["f_start"],
            "step": model.get("f_step", 1.0),
            "stop": model["f_stop"],
        }
    if "f_array" in model:
        parsed = _parse_start_step_stop(model["f_array"])
        model["f_array"] = [float(v) for v in np.atleast_1d(_parse_matlab_range(parsed))]
    if "DomainParam" in model:
        model["DomainParam"] = [[float(v) for v in np.atleast_1d(row)] for row in model["DomainParam"]]
    if "RefDomainParam" in model:
        model["RefDomainParam"] = [[float(v) for v in np.atleast_1d(row)] for row in model["RefDomainParam"]]
    return model


def load_input_param(json_path):
    """Load InputParam from a JSON file.

    The JSON schema intentionally repeats the MATLAB input file fields:
    Config, Model, Advanced, and Mesh. Missing Config/Advanced/Mesh fields
    are filled by the St1 defaults.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fid:
        raw = json.load(fid)
    if "Model" not in raw:
        raise ValueError("JSON model must contain a 'Model' object")
    raw["Model"] = _normalize_arrays(raw["Model"])
    if "Advanced" in raw and "num_eig_max" in raw["Advanced"]:
        raw["Advanced"]["num_eig_max"] = int(raw["Advanced"]["num_eig_max"])
    return as_attrdict(raw), path
