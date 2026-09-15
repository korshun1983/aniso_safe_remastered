"""Plotting helpers, port role of plot_aniso*.m spectrum figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .postproc import build_dispersion_table

# Asymptote line styles: qP - magenta, qSV - black, SH - blue, St - magenta
_ASYMPTOTE_STYLES = [
    ("V_mud", "g", "mud"),
    ("V_qP", "m", "qP"),
    ("V_qSV", "k", "qSV"),
    ("V_SH", "b", "SH"),
    ("V_St", "m", "St"),
]


def _draw_asymptotes(ax, result_dir, kind):
    """Draw horizontal asymptote lines (qP, qSV, SH, St) when available.

    kind="slowness": lines at 0.3048e3/V[km/s] in us/ft;
    kind="velocity": lines at 1e3*V[km/s] in m/s;
    kind="velocity_km_s": lines at V in km/s.
    Velocities are taken from CompStruct.npz written by gen_aniso.
    """
    try:
        from .proc_te import load_compstruct, resolve_data_dir

        comp = load_compstruct(resolve_data_dir(result_dir))
    except Exception:
        return
    asymp = comp.get("Asymp")
    if not asymp:
        return
    drawn = False
    for name, color, label in _ASYMPTOTE_STYLES:
        vel = asymp.get(name)
        if not vel:
            continue
        if kind == "slowness":
            value = 0.3048e3 / vel
        elif kind == "velocity":
            value = 1.0e3 * vel
        else:
            value = vel
        ax.axhline(value, linestyle="--", linewidth=1.5, color=color, label=f"{label} asymptote")
        drawn = True
    if drawn:
        ax.legend(loc="best", fontsize=8)


def _empty_figure(out_png, message):
    """Create an explicit empty figure instead of failing on missing spectra."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def plot_slowness(result_dir, out_png, slowness_limits=None, freqs=None):
    """Plot slowness dispersion curves from Results files."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dispersion_table(result_dir)
    if freqs:
        allowed = {float(v) for v in freqs}
        rows = [row for row in rows if float(row["freq_khz"]) in allowed]
    rows = [row for row in rows if np.isfinite(row["slowness_us_ft"])]
    if slowness_limits and len(slowness_limits) == 2:
        lo, hi = float(slowness_limits[0]), float(slowness_limits[1])
        rows = [row for row in rows if lo <= row["slowness_us_ft"] <= hi]
    if not rows:
        message = (
            "No eigenvalues found in Results files\n"
            f"directory: {Path(result_dir)}\n"
            "Check that --dir points at the gen_aniso output directory\n"
            "(the one that contains Results-*.npz and CompStruct.npz)."
        )
        print("\t" + message.replace("\n", "\n\t"))
        return _empty_figure(out_png, message)

    fig, ax = plt.subplots(figsize=(7, 5))
    by_freq = {}
    for row in rows:
        by_freq.setdefault(row["freq_khz"], []).append(row)
    for freq, items in sorted(by_freq.items()):
        xs = [freq] * len(items)
        ys = [item["slowness_us_ft"] for item in items]
        ax.plot(xs, ys, "o", color="tab:blue", markersize=4)
    _draw_asymptotes(ax, result_dir, "slowness")
    ax.set_xlabel("frequency, kHz")
    ax.set_ylabel("slowness, us/ft")
    ax.grid(True, alpha=0.3)
    ax.set_title("SAFE dispersion: slowness")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def plot_velocity(result_dir, out_png, freqs=None):
    """Plot phase-velocity dispersion curves from Results files."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dispersion_table(result_dir)
    if freqs:
        allowed = {float(v) for v in freqs}
        rows = [row for row in rows if float(row["freq_khz"]) in allowed]
    rows = [row for row in rows if np.isfinite(row["velocity_km_s"])]
    if not rows:
        message = (
            "No eigenvalues found in Results files\n"
            f"directory: {Path(result_dir)}\n"
            "Check that --dir points at the gen_aniso output directory\n"
            "(the one that contains Results-*.npz and CompStruct.npz)."
        )
        print("\t" + message.replace("\n", "\n\t"))
        return _empty_figure(out_png, message)

    fig, ax = plt.subplots(figsize=(7, 5))
    by_freq = {}
    for row in rows:
        by_freq.setdefault(row["freq_khz"], []).append(row)
    for freq, items in sorted(by_freq.items()):
        xs = [freq] * len(items)
        ys = [item["velocity_km_s"] for item in items]
        ax.plot(xs, ys, "o", color="tab:red", markersize=4)
    _draw_asymptotes(ax, result_dir, "velocity_km_s")
    ax.set_xlabel("frequency, kHz")
    ax.set_ylabel("phase velocity, km/s")
    ax.grid(True, alpha=0.3)
    ax.set_title("SAFE dispersion: phase velocity")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png
