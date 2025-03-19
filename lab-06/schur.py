import numpy as np
from numpy.typing import NDArray


def verify_eigenvals(
    A: NDArray[np.float64], eigenvals: NDArray[np.float64], tol: float = 1e-8
) -> bool:
    """
    Verify if the eigenvalues satisfy det(A - λI) ≈ 0 for all λ.
    """
    n = A.shape[0]
    for eigenval in eigenvals:
        det = np.linalg.det(A - eigenval * np.eye(n, dtype=np.float64))
        if np.abs(det) > tol:
            return False
    return True


def verify_schur(
    A: NDArray[np.float64], T: NDArray[np.float64], Z: NDArray[np.float64]
) -> bool:
    """
    Verify the Schur decomposition A ≈ Z @ T @ Z^H.
    """
    return np.allclose(A, Z @ T @ Z.T, atol=1e-8)


def schur(
    A: NDArray[np.float64], max_iter: int = 2_000_00, tol: float = 1e-8
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    QR Algorithm - Computes a Schur decomposition of a matrix A = Z @ T @ Z^H
    https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf

    params
        A - (n, n)

    returns
        T - (n, n) upper triangular with eigenvalues on the diagonal
        Z - (n, n) unitary
    """
    n = A.shape[0]
    T = A.copy().astype(np.float64)
    Z = np.eye(n, dtype=np.float64)
    for iter in range(max_iter):
        Q, R = np.linalg.qr(T)
        T = R @ Q
        Z = Z @ Q

        if verify_eigenvals(A, np.diag(T), tol) and verify_schur(A, T, Z):
            break

    print(f"Converged after {iter} iterations.")

    return T, Z


A = np.array([[5, 0, 2], [9, 3, 8], [1, 6, 7]], dtype=np.float64)
T, Z = schur(A)
print("T:\n", T)
print("Z:\n", Z)
print("Z @ T @ Z^H:\n", Z @ T @ Z.T)

from scipy.linalg import schur

T, Z = schur(A)
print("T:\n", T)
print("Z:\n", Z)
print("Z @ T @ Z^H:\n", Z @ T @ Z.T)
