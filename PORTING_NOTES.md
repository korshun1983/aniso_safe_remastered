# MATLAB -> Python porting notes: Bakken / absorbing domains

## Implemented in this revision

1. **Frequency-dependent `LDomain_in_LSH` geometry**
   - The last physical radius is rebuilt inside every frequency step, as in MATLAB `St3_PrepareBasicMatrices_sp_SAFE.m`.
   - External absorbing-domain radius is also rebuilt per frequency.
   - `InputParam` remains the immutable reference geometry; `CompStruct` is now an independent deep copy, matching MATLAB struct value semantics.

2. **HTTI ABC/PML element kernels**
   - `KM_el_matrix_HTTI_ABC`
   - `KM_el_matrix_HTTI_PML`
   - `KM_el_matrix_HTTI_PML_ABC` (composition of the two supplied transforms; the MATLAB tree references this function but does not contain its source file).

3. **Complex sparse matrix preservation**
   - Element and global assembly no longer casts ABC/PML matrices to `float`.
   - PML stiffness and mass terms retain their imaginary components through `FullMatrices`.

4. **MATLAB `Results-*.mat` loading**
   - `Results.REig_vals` is now read from scipy `mat_struct` objects.
   - A diagonal MATLAB eig-value matrix is converted to a 1-D eigenvalue vector.
   - Eigenvectors are no longer at risk of being selected as eigenvalues by name matching.

5. **Regression inputs and tests**
   - `models/bakken_b_reference_abc.json`
   - `models/bakken_b_reference_noabc.json`
   - frequency geometry tests at 1 and 15 kHz
   - complex ABC/PML/PML+ABC St3 assembly tests
   - MATLAB result-loader regression test

## Bakken geometry control values

For the supplied Bakken solid, `V_SH = 2.169912481961472 km/s`.

| f, kHz | physical outer radius, m | ABC outer radius, m |
|---:|---:|---:|
| 1 | 4.439824963922944 | 6.6097374458844165 |
| 2 | 2.269912481961472 | 3.354868722942208 |
| 4 | 1.1849562409807362 | 1.7274343614711043 |
| 6 | 0.8233041606538241 | 1.1849562409807362 |
| 8 | 0.642478120490368 | 0.913717180735552 |
| 10 | 0.5339824963922944 | 0.7509737445884417 |
| 12 | 0.4616520803269121 | 0.6424781204903681 |
| 15 | 0.3893216642615296 | 0.5339824963922943 |

## MATLAB-reference quirks preserved deliberately

The goal of the default implementation is numerical comparability with the supplied MATLAB source, including its observable behavior:

- `KM_el_matrix_HTTI_ABC.m` checks `ABC_account_r == 1`, while Bakken sets `ABC_account_r='yes'`. In the supplied MATLAB code this means the `1/r` branch is not activated. Python mirrors that behavior.
- `KM_el_matrix_HTTI_ABC.m` multiplies the entire `CijMatrix` inside the interpolation-node loop. Therefore the scale at node `k` is cumulative. Python mirrors this behavior.
- Internal rectangular PML in MATLAB computes `y_zv` from `DomainRx(end)`, not `DomainRy(end)`. Python currently mirrors that source line.
- MATLAB references `KM_el_matrix_HTTI_PML_ABC` but that file is absent from the supplied tree. The Python combined kernel is consequently inferred rather than source-verifiable.

These can later be separated into `matlab_compat` and `corrected_physics` modes if desired.

## Remaining major parity gap

The Python mesher is **not** a port of MATLAB Mesh2D v2.4. MATLAB uses `meshfaces` with quadtree size control, boundary re-discretization, smoothing and face-by-face meshing. Python currently uses deterministic concentric rings plus `scipy.spatial.Delaunay` and does not reproduce Mesh2D `hmax/dhmax` behavior. This is now the largest known obstacle to close eigenvalue-by-eigenvalue parity after the physics/geometry fixes.

## Verification performed

- Python convolution matrices were compared with the supplied MATLAB `.mat` references:
  - `NNNConvMatrixInt`
  - `dNNNConvMatrixInt`
  - `NNdNConvMatrixInt`
  - `dNNdNConvMatrixInt`
  - `NENENEConvMatrixInt`
- Maximum relative differences were approximately `1e-13 ... 1e-14`.
- All 8 requested Bakken frequencies passed St3 for both ABC and no-ABC configurations.
- Reduced-mesh end-to-end runs passed St3+St4 for `abc`, `pml`, and `pml+abc` and returned complex wavenumbers.
- Test suite: `7 passed`.

The uploaded archives did not contain the user's saved Bakken MATLAB `Results-*.mat` eigenvalue files, so direct numerical matching of those eigenvalue sets could not yet be performed.
