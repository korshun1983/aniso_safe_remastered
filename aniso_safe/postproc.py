"""Post-processing helpers for Results files, port role of proc_aniso_TE.m."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class DispersionRecord:
    """One frequency point with its eigenvalues."""

    freq_khz: float
    eigenvalues: np.ndarray
    source: Path


def _freq_from_name(path):
    """Extract frequency from names like Results-1.5.npz or Results-1.5.mat."""
    m = re.search(r"Results-([+-]?\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else np.nan


def _walk_matlab_struct(value):
    """Yield nested scipy MATLAB structs and arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            yield value
            for name in value.dtype.names:
                yield from _walk_matlab_struct(value[name])
        elif value.dtype == object:
            for item in value.ravel():
                yield from _walk_matlab_struct(item)
    elif isinstance(value, np.void) and value.dtype.names:
        yield value
        for name in value.dtype.names:
            yield from _walk_matlab_struct(value[name])
    elif hasattr(value, "_fieldnames"):
        yield value
        for name in value._fieldnames:
            yield from _walk_matlab_struct(getattr(value, name))


def _normalize_eigenvalue_array(value):
    """Convert MATLAB/Python eig output to the 1-D eigenvalue vector."""
    arr = np.asarray(value)
    if not arr.size or not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.asarray(arr, dtype=complex).squeeze()
    # MATLAB eigs returns REig_vals as a diagonal k-by-k matrix, while the
    # Python solver stores a vector.  Never flatten the off-diagonal zeros.
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        arr = np.diag(arr)
    return arr.ravel()


def _extract_eigenvalues(obj):
    """Find the eigenvalue array without mistaking eigenvectors for values."""
    preferred = ("eigenvalues", "REig_vals", "Eig_vals", "REig_Lvals")

    if isinstance(obj, dict):
        for wanted in preferred:
            for key, value in obj.items():
                if str(key).lower() == wanted.lower():
                    arr = _normalize_eigenvalue_array(value)
                    if arr is not None:
                        return arr
        for value in obj.values():
            arr = _extract_eigenvalues(value)
            if arr is not None:
                return arr
        return None

    if hasattr(obj, "_fieldnames"):
        fields = list(obj._fieldnames)
        for wanted in preferred:
            for name in fields:
                if name.lower() == wanted.lower():
                    arr = _normalize_eigenvalue_array(getattr(obj, name))
                    if arr is not None:
                        return arr
        for name in fields:
            arr = _extract_eigenvalues(getattr(obj, name))
            if arr is not None:
                return arr
        return None

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for wanted in preferred:
                for name in obj.dtype.names:
                    if name.lower() == wanted.lower():
                        arr = _normalize_eigenvalue_array(obj[name])
                        if arr is not None:
                            return arr
            for name in obj.dtype.names:
                arr = _extract_eigenvalues(obj[name])
                if arr is not None:
                    return arr
        elif obj.dtype == object:
            for item in obj.ravel():
                arr = _extract_eigenvalues(item)
                if arr is not None:
                    return arr
    return None

def load_result_file(path):
    """Load one Results file from NPZ or MAT format."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        payload = {key: data[key] for key in data.files}
        eigenvalues = _extract_eigenvalues(payload)
        return DispersionRecord(_freq_from_name(path), eigenvalues if eigenvalues is not None else np.zeros(0, complex), path)
    if path.suffix.lower() == ".mat":
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        eigenvalues = _extract_eigenvalues(data)
        return DispersionRecord(_freq_from_name(path), eigenvalues if eigenvalues is not None else np.zeros(0, complex), path)
    raise ValueError(f"Unsupported result file: {path}")


def discover_results(result_dir):
    """Find and load Results files sorted by frequency.

    Accepts the output directory itself or its parent (with the data one
    level below, e.g. from an older layout that used output/ or results/).
    """
    result_dir = Path(result_dir)
    files = sorted(list(result_dir.glob("Results-*.npz")) + list(result_dir.glob("Results-*.mat")))
    if not files:
        for sub in ("output", "results"):
            files = sorted(
                list((result_dir / sub).glob("Results-*.npz"))
                + list((result_dir / sub).glob("Results-*.mat"))
            )
            if files:
                print(f"\tUsing data directory: {result_dir / sub}")
                break
    records = [load_result_file(path) for path in files]
    return sorted(records, key=lambda rec: rec.freq_khz)


def phase_velocity_km_s(freq_khz, eigenvalues):
    """Convert wavenumber eigenvalues to phase velocity in km/s."""
    k = np.asarray(eigenvalues, dtype=complex)
    omega = 2.0 * np.pi * float(freq_khz) * 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity_m_s = omega / k
    return velocity_m_s.real / 1000.0


def slowness_us_ft(freq_khz, eigenvalues):
    """Convert wavenumber eigenvalues to slowness in us/ft."""
    velocity_m_s = phase_velocity_km_s(freq_khz, eigenvalues) * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        slowness = 1e6 * 0.3048 / velocity_m_s
    return slowness.real


def build_dispersion_table(result_dir):
    """Build frequency/eigenvalue/slowness table for plotting."""
    rows = []
    for rec in discover_results(result_dir):
        for eig in rec.eigenvalues:
            rows.append(
                {
                    "freq_khz": rec.freq_khz,
                    "eigenvalue": eig,
                    "velocity_km_s": phase_velocity_km_s(rec.freq_khz, eig),
                    "slowness_us_ft": slowness_us_ft(rec.freq_khz, eig),
                    "source": str(rec.source),
                }
            )
    return rows


def save_slowness_npz(result_dir, out_file=None):
    """Save a MATLAB-like Slowness summary in NPZ form."""
    result_dir = Path(result_dir)
    out_file = Path(out_file) if out_file else result_dir / "Slowness.npz"
    rows = build_dispersion_table(result_dir)
    freq = np.array([row["freq_khz"] for row in rows], dtype=float)
    eig = np.array([row["eigenvalue"] for row in rows], dtype=complex)
    vel = np.array([row["velocity_km_s"] for row in rows], dtype=float)
    slo = np.array([row["slowness_us_ft"] for row in rows], dtype=float)
    np.savez_compressed(out_file, freq_khz=freq, eigenvalues=eig, velocity_km_s=vel, slowness_us_ft=slo)
    return out_file, rows
