#!/usr/bin/env python3
"""Process raw gen_aniso results into LayerTE and Slowness files.

CLI wrapper around aniso_safe.proc_te.proc_aniso_TE, port role of
proc_aniso_TE.m.
"""

from __future__ import annotations

import argparse

from aniso_safe.proc_te import proc_aniso_TE


def main():
    parser = argparse.ArgumentParser(description="Process ANISO SAFE results (kinetic energy chain)")
    parser.add_argument("--dir", required=True, help="Directory with CompStruct.npz, FEMatrices-*.npz and Results-*.npz")
    parser.add_argument("--write-result", default="y", choices=["y", "n"], help="Save (y) or not (n) LayerTE files")
    parser.add_argument("--dtheta-inc", type=int, default=3, help="Decrease azimuthal step DTheta this many times")
    parser.add_argument("--nharm", type=int, default=4, help="Number of trigonometric Fourier harmonics")
    parser.add_argument("--polar-grid", default="n", choices=["y", "n"], help="Plot polar grid with mesh for each layer")
    args = parser.parse_args()

    proc_aniso_TE(
        args.dir,
        write_result=args.write_result,
        DThetaInc=args.dtheta_inc,
        NHarm=args.nharm,
        Polar_Grid=args.polar_grid,
    )


if __name__ == "__main__":
    main()
