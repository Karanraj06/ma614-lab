import numpy as np
from numpy.typing import NDArray


def lu_decomposition(
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Performs LU Decomposition (A = LU)"""
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix must be square.")

    L = np.eye(n, dtype=np.float64)
    U = A.copy()

    for i in range(n):
        for j in range(i + 1, n):
            if U[i, i] == 0:
                raise ValueError("Singular matrix detected.")

            c = U[j, i] / U[i, i]
            U[j, i:] -= c * U[i, i:]
            L[j, i] = c

    return L, U


def forward_substitution(
    L: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solves Ly = b for y"""
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y


def back_substitution(
    U: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solves Ux = y for x"""
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x


def ldv_decomposition(A: NDArray[np.float64]):
    """Performs LDV Decomposition (A = LDV)"""
    L, U = lu_decomposition(A)
    D = np.diag(np.diag(U))
    V = np.zeros_like(A, dtype=np.float64)

    for i in range(A.shape[0]):
        if D[i, i] == 0:
            raise ValueError("Singular matrix detected.")
        V[i, :] = U[i, :] / D[i, i]

    return L, D, V


A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=np.float64)
b = np.array([8, -11, -3], dtype=np.float64)

L, U = lu_decomposition(A)
y = forward_substitution(L, b)
x = back_substitution(U, y)

print("L:\n", L)
print("U:\n", U)
print("Solution x:", x)

L_ldv, D_ldv, V_ldv = ldv_decomposition(A)
print("L (LDV):\n", L_ldv)
print("D:\n", D_ldv)
print("V:\n", V_ldv)
