import numpy as np

from numpy.typing import NDArray
from sympy import Matrix


def schur_complement(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    D: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the Schur complement of D in the block matrix M
    (M|D) = A - B @ D^-1 @ C
    """
    if np.linalg.det(D) == 0:
        raise ValueError("D must be invertible.")

    return A - B @ np.linalg.inv(D) @ C


A = np.array([[4, 1], [1, 3]])
B = np.array([[2, 0], [0, 1]])
C = np.array([[1, 3], [1, 2]])
D = np.array([[3, 2], [2, 1]])
print(f"Schur complement of D in M:\n{schur_complement(A, B, C, D)}\n")

# Eigenvalues and eigenvectors
A = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 1]], dtype=np.float64)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues of A:\n{eigenvalues}\n")
print(f"Eigenvectors of A:\n{eigenvectors}\n")

# Diagonalization A = PDP^-1
P = eigenvectors
D = np.diag(eigenvalues)
print(np.allclose(A, P @ D @ np.linalg.inv(P)))

# Diagonalization using sympy
A = Matrix([[2, 3, 1], [4, 5, 1], [6, 7, 1]])
eigenvalues = A.eigenvals()
eigenvectors = A.eigenvects()
P, D = A.diagonalize()

P_np = np.array(P.evalf(), dtype=np.float64)
D_np = np.array(D.evalf(), dtype=np.float64)
P_inv_np = np.array(P.inv().evalf(), dtype=np.float64)
A_np = np.array(A.evalf(), dtype=np.float64)
print(np.allclose(A_np, P_np @ D_np @ P_inv_np))
