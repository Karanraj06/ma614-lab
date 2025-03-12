import numpy as np
from numpy.typing import NDArray


def qr(
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    m, n = A.shape
    Q = np.eye(m, dtype=np.float64)
    R = A.copy().astype(np.float64)

    for k in range(n):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * np.sign(x[0] if x[0] != 0 else 1)
        v = x + e
        v /= np.linalg.norm(v)

        H_k = np.eye(m, dtype=np.float64)
        H_k[k:, k:] -= 2 * np.outer(v, v)

        R = H_k @ R
        Q = Q @ H_k.T

    return Q, R


A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float64)
Q, R = qr(A)
print("Q:\n", Q)
print("R:\n", R)
print("Q @ R:\n", Q @ R)
