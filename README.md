# ANISO_SAFE Python port

This project is a faithful step-by-step Python port of the MATLAB `matlab_test_code` repository. The public function names intentionally follow the MATLAB stages: `St1_SetModel`, `St2_PrepareModel_sp_SAFE`, `PrepareMeshBH`, `AddNodesCubic`, `FindBEdges`, and `gen_aniso`.

## Current scope

- JSON physical model loading (`models/*.json`) with MATLAB-like range syntax such as `0.5:0.25:15`.
- Step 1 initialization: `Config`, `Methods`, user `Model`, and `Advanced` defaults.
- Step 2 preparation: unit conversion factors, domain count, per-domain variable numbers, exact dispersion-curve asymptotes (`ComputeAsymptotesSAFE` + `V_phase_VTI_exact_RPH`: `V_mud`, `V_qP`, `V_qSV`, `V_SH`, `V_St` in km/s), and method-name registry.
- Mesh control path: boundary construction for circular/rectangular outer shapes, Delaunay triangulation, domain tagging, cubic node enrichment, and PNG mesh output controlled by `Mesh.output`.
- Step 3 matrix chain (`aniso_safe/St3_1_PrepareBasicMatrices_sp_SAFE.py`, `aniso_safe/basic.py`, `aniso_safe/fematrices.py`): basic matrices and convolution integrals, per-domain stiffness/mass blocks for `fluid` and `HTTI` domains, fluid–solid (`FS`) and `ff`/`ss` interface matrices, `rigid`/`free` outer boundary conditions, full matrix assembly, and removal of redundant variables.
- Step 4 eigensolver (`aniso_safe/St4_ComputeSolution_sp_SAFE.py`): the SAFE polynomial eigenvalue problem in `k` with the MATLAB linearization, solved by shift-invert `scipy.sparse.linalg.eigs` near `omega/EigSearchStart`.
- TE post-processing (`aniso_safe/proc_te.py`, port of `proc_aniso_TE.m` + `St61_Proc_Fluid/Solid_TE_sp_SAFE.m`): polar (r, theta) grid per layer, displacement interpolation, radial kinetic-energy density, azimuthal Fourier decomposition; writes `LayerTE-<freq>-<layer>.npz` and `Slowness.npz`.
- Mode classification and TE-based dispersion plots (`aniso_safe/intr_te.py`, port of `intr_aniso_TE.m`): monopole / flexural ± / quadrupole ± / 3pole ± classification, `changer_12` swap, full slowness/velocity figures and per-mode-class PNG figures with the qP/qSV/SH/St asymptote lines. The plain dispersion figures (`plot_aniso_TE2.py`, `plot_aniso.py`) draw the same asymptotes when `CompStruct.npz` is present.
- Orchestration and persistence (`aniso_safe/gen_aniso.py`, port of `gen_aniso.m`): `FEMatrices-<freq>.npz`, `Results-<freq>.npz`, `CompStruct.npz`, and `mesh-<freq>.png` when mesh output is enabled.

The dispersion curves are computed by Python itself. `Advanced.EigSearchStart` is the phase velocity (km/s) near which the eigensolver searches for modes; pick it close to the velocities of interest (e.g. `1.5` for fluid-dominated modes). The PML/ABC additional-domain kernels (`KM_el_matrix_HTTI_PML/ABC`, `MatricesPartsPML/ABC`) and the frequency-dependent additional-domain geometry (`LDomain_in_LSH == 'yes'`) are still pending.

## Project structure

```
aniso_safe_python/
├── gen_aniso.py                 # CLI: main run, port of gen_aniso.m
├── proc_aniso_TE.py             # CLI: TE post-processing, port of proc_aniso_TE.m
├── plot_aniso_TE.py             # CLI: mode classification + mode plots, port of intr_aniso_TE.m
├── plot_aniso_TE2.py            # CLI: plain slowness/velocity curves from Results-*.npz
├── plot_aniso.py                # CLI: dispersion curves, port of plot_aniso.m
├── aniso_safe/
│   ├── gen_aniso.py                        # orchestration, port of gen_aniso.m
│   ├── St1_SetModel.py                     # port of St1_SetModel.m
│   ├── St2_PrepareModel_sp_SAFE.py         # port of St2_*_sp_SAFE.m
│   ├── St3_1_PrepareBasicMatrices_sp_SAFE.py  # port of St3_1_PrepareBasicMatrices_sp_SAFE.m
│   ├── St4_ComputeSolution_sp_SAFE.py      # port of St4_ComputeSolution_sp_SAFE.m
│   ├── basic.py                   # AssembleBasicMatrices_sp_SAFE, convolution integrals
│   ├── fematrices.py              # FEmatrices/*.m: element matrices, assembly, BCs
│   ├── mesh.py                    # mesh/*.m: PrepareMeshBH, AddNodesCubic, FindBEdges, ...
│   ├── proc_te.py                 # proc_aniso_TE.m + St61_Proc_Fluid/Solid_TE_sp_SAFE.m
│   ├── intr_te.py                 # intr_aniso_TE.m
│   ├── io_json.py                 # JSON model loading
│   ├── postproc.py                # Results file reading helpers
│   ├── plotting.py                # simple dispersion figures
│   └── structures.py              # AttrDict helpers
├── models/                      # JSON physical models
├── examples/
└── tests/
```

## Run

```bash
cd aniso_safe_python
python3 gen_aniso.py --model models/bakken_b_00.json --output-dir output
```

All artifacts (`CompStruct.npz`, `FEMatrices-<freq>.npz`, `Results-<freq>.npz`,
`mesh-<freq>.png`) are written directly into the directory given by
`--output-dir` (default: the model path without the `.json` extension).
Pass this same directory as `--dir` to every post-processing script below.
As a fallback the post-processing tools also look one level below into
`output/` and `results/` subdirectories.

## Spectrum post-processing and plotting

The MATLAB post-processing entry points have Python analogs:

```bash
# LayerTE + Slowness files (kinetic energy chain)
python3 proc_aniso_TE.py --dir output

# mode classification and per-mode dispersion figures
python3 plot_aniso_TE.py --dir output --velocity y

# plain dispersion curves straight from the eigenvalues
python3 plot_aniso_TE2.py --dir output --out dispersion.png --velocity-out velocity.png
python3 plot_aniso.py --dir output --slowness y
```

`plot_aniso_TE2.py` reads `Results-*.npz` and MATLAB `Results-*.mat`. If a result file contains no eigenvalues, it writes an explicit empty figure instead of failing.

## Simple verification example

`models/example_water_hard_pipe.json` is the minimal sanity check: a water-filled
circular pipe (radius 0.1 m, c = 1500 m/s, rho = 1 g/cm3) with a single fluid
domain and `BCType = ["free"]`.

Important boundary-condition note: in this SAFE formulation `rigid` removes all
boundary-node variables, which for a fluid domain imposes p = 0 (a
pressure-release, i.e. acoustically soft wall). The natural `free` condition
does nothing at the boundary, which for a fluid is the acoustically hard wall
(dp/dn = 0). So a hard-wall water pipe is `free`, not `rigid`.

Run the full chain:

```bash
python3 gen_aniso.py --model models/example_water_hard_pipe.json --output-dir output_hard --mesh-output yes
python3 proc_aniso_TE.py --dir output_hard
python3 plot_aniso_TE.py --dir output_hard --velocity y
```

Analytic reference for the hard-wall pipe (a = 0.1 m, c = 1500 m/s,
kz = sqrt((w/c)^2 - (x'_mn/a)^2)):

| mode                | behaviour                                        |
|---------------------|--------------------------------------------------|
| fundamental (m = 0) | non-dispersive, v = 1500 m/s (203.2 us/ft) flat  |
| flexural (m = 1)    | cut-on f = 1.8412*c/(2*pi*a) = 4.40 kHz          |
| quadrupole (m = 2)  | cut-on f = 3.0542*c/(2*pi*a) = 7.29 kHz          |
| radial (m = 0)      | cut-on f = 3.8317*c/(2*pi*a) = 9.15 kHz          |

The port reproduces the flat fundamental at exactly 1.500 km/s and the
dispersive flexural/quadrupole branches with the analytic cut-on frequencies
(the `modes_*.png` figures). Modes far above the shift-invert target
(`Advanced.EigSearchStart`, in km/s) are simply not returned by `eigs`; that is
expected solver behaviour, not an error.

## JSON model schema

The JSON root mirrors the MATLAB input file:

- `Model`: geometry (`DomainRx`, `DomainRy`, `DomainTheta`, `DomainEcc`, `DomainEccAngle`), `DomainType`, `DomainParam`, `RefDomainType`, `RefDomainParam`, `BCType`, `f_array`, `DomainNth`, `mud_domain`, PML/ABC and additional-domain fields.
- `Advanced`: `num_eig_max`, `EigSearchStart`, `VisualizeMesh`, `N_nodes`, `NEdge_nodes`.
- `Mesh`: `output` (`yes`/`no`) and `ext_boundary_shape` (`cir`/`rec`).

`Model.f_array` accepts three equivalent forms:

```json
"f_array": [0.5, 1.0, 2.0, 4.0]
```

```json
"f_array": "0.5:0.25:15"
```

```json
"f_array": {"start": 0.5, "step": 0.25, "stop": 15}
```

It also accepts separate fields when `f_array` is absent:

```json
"f_start": 0.5,
"f_step": 0.25,
"f_stop": 15
```

All code comments are English only by project rule.
