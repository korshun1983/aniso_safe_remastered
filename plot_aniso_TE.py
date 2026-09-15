#!/usr/bin/env python3
"""Classify modes by kinetic energy and plot the dispersion curves.

CLI wrapper around aniso_safe.intr_te.intr_aniso_TE, port role of
intr_aniso_TE.m.
"""

from __future__ import annotations

import argparse

from aniso_safe.intr_te import intr_aniso_TE


def _parse_floats(text):
    if text is None:
        return None
    return [float(v) for v in text.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser(description="Classify ANISO SAFE modes and plot dispersion curves")
    parser.add_argument("--dir", required=True, help="Directory with LayerTE-*.npz and Slowness.npz")
    parser.add_argument("--freqs", default=None, help="Space/comma separated frequency list in kHz (default: all)")
    parser.add_argument("--num-eigs-class-show", type=int, default=2, help="Number of classified eigs to plot")
    parser.add_argument("--changer-12", default="y", choices=["y", "n"], help="Swap mode 1/2 across the qSV line")
    parser.add_argument("--max-harm-limit", type=float, default=0.1, help="Harmonic limit for symmetry classification")
    parser.add_argument(
        "--show-modes",
        default="y y y y n n n",
        help="Seven y/n flags: monopole, flexural +/-, quadrupole +/-, 3pole +/-",
    )
    parser.add_argument("--slowness", default="y", choices=["y", "n"], help="Plot the full slowness figure")
    parser.add_argument("--velocity", default="n", choices=["y", "n"], help="Plot the full velocity figure")
    parser.add_argument("--slowness-limits", nargs=2, type=float, default=(100.0, 250.0), help="Slowness range in us/ft")
    parser.add_argument("--velocity-limits", nargs=2, type=float, default=(1300.0, 3200.0), help="Velocity range in m/s")
    parser.add_argument("--out", default=None, help="Output directory for the PNG figures (default: --dir)")
    args = parser.parse_args()

    show_modes = tuple(v.strip().lower() for v in args.show_modes.replace(",", " ").split())
    if len(show_modes) != 7:
        parser.error("--show-modes must contain exactly seven y/n flags")

    result = intr_aniso_TE(
        args.dir,
        freqs=_parse_floats(args.freqs),
        num_eigs_class_show=args.num_eigs_class_show,
        changer_12=args.changer_12,
        max_harm_limit=args.max_harm_limit,
        show_modes=show_modes,
        slowness=args.slowness,
        velocity=args.velocity,
        slowness_limits=tuple(args.slowness_limits),
        velocity_limits=tuple(args.velocity_limits),
        out_dir=args.out,
    )
    if result is not None:
        for fig in result.figures:
            print(f"\tSaved: {fig}")


if __name__ == "__main__":
    main()
