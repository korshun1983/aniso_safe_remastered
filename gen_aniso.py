#!/usr/bin/env python3
"""Command line entry point for the Python port of gen_aniso.m."""

from __future__ import annotations

import argparse
from pathlib import Path

from aniso_safe.gen_aniso import gen_aniso


def main():
    parser = argparse.ArgumentParser(description="Run ANISO SAFE Python workflow")
    parser.add_argument("--model", required=True, help="Path to the JSON physical model")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")
    parser.add_argument("--mesh-output", choices=["yes", "no"], default=None, help="Override Mesh.output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    comp_struct, saved_files = gen_aniso(
        args.model, output_dir=output_dir, mesh_output=args.mesh_output
    )
    print("Saved files:")
    for path in saved_files:
        print(f"\t{path}")
    return comp_struct


if __name__ == "__main__":
    main()
