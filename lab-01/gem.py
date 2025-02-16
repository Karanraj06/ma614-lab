import numpy as np


def gaussian_elimination(
    A: np.ndarray[np.float64], b: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """
    assumptions
        rank(A) = rank([A|B]) = n (number of unknowns/columns) - unique solution

    params
        A - (m, n)
        b - (m,)

    returns
        x - (n,)
    """
    m, n = A.shape
    aug = np.hstack((A, b.reshape(-1, 1)))
    for i in range(min(m, n)):  # Forward Elimination
        pivot_row_idx = np.argmax(np.abs(aug[i:, i])) + i
        if np.abs(aug[pivot_row_idx, i]) < 1e-12:
            raise ValueError("System doesn't have a unique solution.")

        if pivot_row_idx != i:
            aug[[i, pivot_row_idx]] = aug[[pivot_row_idx, i]]

        pivot = aug[i, i]
        aug[i, :] /= pivot

        for j in range(i + 1, m):
            aug[j, :] -= aug[j, i] * aug[i, :]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):  # Back Substitution
        if i >= m or aug[i, i] == 0:
            continue

        x[i] = aug[i, -1] - np.dot(aug[i, i + 1 : n], x[i + 1 :])

    return x


A = np.array([[1, -1, 1], [2, 1, 8], [4, 2, -3]], dtype=np.float64)
b = np.array([3, 18, -2], dtype=np.float64)
x = gaussian_elimination(A, b)
print(x)
