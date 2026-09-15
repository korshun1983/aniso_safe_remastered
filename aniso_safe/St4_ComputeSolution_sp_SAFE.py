"""Generalized eigenvalue solution, port of St4_ComputeSolution_sp_SAFE.m."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, splu

from .structures import AttrDict


def _as_sparse(matrix):
    """Convert a matrix to CSR without changing its numerical content."""
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def _filter_by_residual(amatrix, bmatrix, eig_vals, eig_vecs, res_tol):
    """Keep only the Ritz pairs with a small relative backward error.

    For the strongly non-normal SAFE pencil (fluid-solid coupling makes the
    matrix blocks differ by many orders of magnitude) ARPACK may return
    spurious Ritz values.  The check is done block-wise: the first block row
    of the linearization requires z2 = k*z1, and the second block row is the
    quadratic pencil itself.  A global residual norm is NOT reliable here,
    because the huge stiffness-block entries mask the first-row inconsistency.
    """
    if eig_vals.size == 0:
        return eig_vals, eig_vecs
    n = amatrix.shape[0] // 2
    z1 = eig_vecs[:n, :]
    z2 = eig_vecs[n:, :]
    az2 = (amatrix @ eig_vecs)[n:, :]
    bz2 = (bmatrix @ eig_vecs)[n:, :]
    # first block row: z2 = k * z1
    r1 = np.linalg.norm(z2 - eig_vals[None, :] * z1, axis=0) / (
        np.linalg.norm(z2, axis=0) + np.abs(eig_vals) * np.linalg.norm(z1, axis=0) + 1e-300
    )
    # second block row: the quadratic pencil itself
    r2 = np.linalg.norm(az2 - eig_vals[None, :] * bz2, axis=0) / (
        np.linalg.norm(az2, axis=0) + np.abs(eig_vals) * np.linalg.norm(bz2, axis=0) + 1e-300
    )
    keep = (r1 < res_tol) & (r2 < res_tol)
    return eig_vals[keep], eig_vecs[:, keep]


def _shift_invert_refined(amatrix, bmatrix, sigma, n_eig, eig_tol, n_refine):
    """Shift-invert Arnoldi on OP = (A - sigma*B)^{-1} B with iterative refinement.

    The pencil blocks have a huge dynamic range (fluid vs solid scales), so a
    single SuperLU solve inside ARPACK loses the eigen-directions closest to
    the shift.  Refining the solve (x += solve(b - (A-sB) x)) restores the
    forward accuracy and lets ARPACK converge to the true eigenvalues near
    sigma.  Eigenvalues are recovered as k = sigma + 1/theta.
    """
    asb = (amatrix - sigma * bmatrix).tocsc()
    lu = splu(asb)

    def _matvec(x):
        rhs = bmatrix @ x
        y = lu.solve(rhs)
        for _ in range(n_refine):
            y = y + lu.solve(rhs - asb @ y)
        return y

    op = LinearOperator(amatrix.shape, matvec=_matvec, dtype=complex)
    theta, zz = eigs(op, k=n_eig, which="LM", tol=eig_tol)
    return sigma + 1.0 / theta, zz


def solve_polynomial_evp_k(FullMatrices, freq_khz, num_eig_max, eig_search_start, eig_tol=1e-8,
                           res_tol=1.0e-4, n_refine=2):
    """Solve the SAFE polynomial EVP for k at a fixed frequency.

    MATLAB statement:
        K1 + 1i*k*K2 + k^2*K3 - omega^2*M - 1i*omega*P
    Linearization used in St4_ComputeSolution_sp_SAFE.m:
        A = [0 I; -K -1i*K2], B = [I 0; 0 K3]
        K = K1 - omega_var_1000^2*M + 1i*omega_var_1000*P

    Numerical note (difference from the MATLAB black-box eigs call):
    the returned Ritz pairs are filtered by their relative backward error,
    and if too few pairs survive, the computation is retried with an
    iteratively refined shift-invert operator.  This does not change the
    problem statement, only the reliability of the sparse eigensolver.
    """
    required = ("K1Matrix", "K2Matrix", "K3Matrix", "MMatrix", "PMatrix")
    if not all(name in FullMatrices for name in required):
        return AttrDict(
            REig_vals=np.zeros((0,), dtype=complex),
            REig_vecs=np.zeros((0, 0), dtype=complex),
            omega_val=2.0 * np.pi * float(freq_khz),
            status="pending FullMatrices",
        )

    k1 = _as_sparse(FullMatrices.K1Matrix)
    k2 = _as_sparse(FullMatrices.K2Matrix)
    k3 = _as_sparse(FullMatrices.K3Matrix)
    mass = _as_sparse(FullMatrices.MMatrix)
    pmat = _as_sparse(FullMatrices.PMatrix)

    omega_val = 2.0 * np.pi * float(freq_khz)
    omega_var_1000 = omega_val * 1.0e3
    k_matrix = k1 - (omega_var_1000**2) * mass + (omega_var_1000 * 1j) * pmat

    n = k1.shape[0]
    zm = sparse.csr_matrix((n, n), dtype=complex)
    eye = sparse.identity(n, dtype=complex, format="csr")
    amatrix = sparse.bmat([[zm, eye], [-k_matrix, -1j * k2]], format="csr")
    bmatrix = sparse.bmat([[eye, zm], [zm, k3]], format="csr")

    k_start_val = omega_val / float(eig_search_start)
    n_eig = int(max(1, min(int(num_eig_max), 2 * n - 2)))
    status = "ok"
    try:
        eig_vals, eig_vecs = eigs(
            amatrix,
            M=bmatrix,
            k=n_eig,
            sigma=k_start_val,
            tol=float(eig_tol),
            which="LM",
        )
        eig_vals, eig_vecs = _filter_by_residual(
            amatrix, bmatrix, eig_vals, eig_vecs, res_tol
        )
    except Exception:
        eig_vals = np.zeros((0,), dtype=complex)
        eig_vecs = np.zeros((0, 0), dtype=complex)

    # Retry with the refined shift-invert operator when the plain ARPACK run
    # did not produce enough converged eigenvalues
    if eig_vals.size < n_eig:
        try:
            r_vals, r_vecs = _shift_invert_refined(
                amatrix, bmatrix, k_start_val, n_eig, float(eig_tol), int(n_refine)
            )
            r_vals, r_vecs = _filter_by_residual(
                amatrix, bmatrix, r_vals, r_vecs, res_tol
            )
            # keep the better of the two attempts
            if r_vals.size > eig_vals.size:
                eig_vals, eig_vecs = r_vals, r_vecs
                status = "ok (refined shift-invert)"
        except Exception as exc:
            if eig_vals.size == 0:
                return AttrDict(
                    REig_vals=np.zeros((0,), dtype=complex),
                    REig_vecs=np.zeros((0, 0), dtype=complex),
                    omega_val=omega_val,
                    status=f"eigs failed: {exc}",
                )

    if eig_vals.size == 0:
        return AttrDict(
            REig_vals=np.zeros((0,), dtype=complex),
            REig_vecs=np.zeros((0, 0), dtype=complex),
            omega_val=omega_val,
            status="no converged eigenvalues",
        )

    order = np.argsort(np.abs(eig_vals))[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]
    return AttrDict(REig_vals=eig_vals, REig_vecs=eig_vecs, omega_val=omega_val, status=status)


def St4_ComputeSolution_sp_SAFE(CompStruct, BasicMatrices, FEMatrices, FullMatrices):
    """Compute SAFE spectrum with the MATLAB function contract."""
    freq = float(CompStruct.f_grid[CompStruct.if_grid])
    solved = solve_polynomial_evp_k(
        FullMatrices,
        freq,
        CompStruct.Advanced.num_eig_max,
        CompStruct.Advanced.EigSearchStart,
        CompStruct.Advanced.EigsOptions.tol,
    )
    return AttrDict(
        freq_khz=freq,
        eigenvalues=solved.REig_vals,
        eigenvectors=solved.REig_vecs,
        omega_val=solved.omega_val,
        status=solved.status,
    )
