# ANISO_SAFE Python port

This project is a faithful step-by-step Python port of the MATLAB `matlab_test_code` repository. The public function names intentionally follow the MATLAB stages: `St1_SetModel`, `St2_PrepareModel_sp_SAFE`, `PrepareMeshBH`, `AddNodesCubic`, `FindBEdges`, and `gen_aniso`.

## Current scope

- JSON physical model loading (`models/*.json`) with MATLAB-like range syntax such as `0.5:0.25:15`.
- Step 1 initialization: `Config`, `Methods`, user `Model`, and `Advanced` defaults.
- Step 2 preparation: unit conversion factors, domain count, per-domain variable numbers, reference asymptote estimates, and method-name registry.
- Mesh control path: boundary construction for circular/rectangular outer shapes, Delaunay triangulation, domain tagging, cubic node enrichment, and PNG mesh output controlled by `Mesh.output`.
- Orchestration and persistence: `FEMatrices-<freq>.npz`, `Results-<freq>.npz`, `CompStruct.npz`, and `mesh-<freq>.png` when mesh output is enabled.

The sparse SAFE matrix kernels and eigensolver are isolated behind `St3_PrepareBasicMatrices_sp_SAFE` and `St4_ComputeSolution_sp_SAFE`. Those functions keep the MATLAB call contract and currently return explicit pending status instead of fabricated spectra.

## Run

```bash
cd /mnt/agents/output/aniso_safe_python
python3 gen_aniso.py --model models/bakken_b_00.json
```

Mesh PNG files are written to `models/bakken_b_00/output/mesh-<freq>.png` and copied to `models/bakken_b_00/results/`.

## JSON model schema

The JSON root mirrors the MATLAB input file:

- `Model`: geometry (`DomainRx`, `DomainRy`, `DomainTheta`, `DomainEcc`, `DomainEccAngle`), `DomainType`, `DomainParam`, `RefDomainType`, `RefDomainParam`, `BCType`, `f_array`, `DomainNth`, `mud_domain`, PML/ABC and additional-domain fields.
- `Advanced`: `num_eig_max`, `EigSearchStart`, `VisualizeMesh`, `N_nodes`, `NEdge_nodes`.
- `Mesh`: `output` (`yes`/`no`) and `ext_boundary_shape` (`cir`/`rec`).

All code comments are English only by project rule.
