import numpy as np
from numpy.typing import NDArray


def cholesky(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assumptions - A is symmetric and positive definite"""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            s = np.sum(L[i, :j] * L[j, :j])

            if i == j:
                L[i, i] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)
L = cholesky(A)
print(L)
print(np.allclose(L @ L.T, A))
