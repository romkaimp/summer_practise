import numpy as np

import logging
from typing import Any


infoLogger = logging.getLogger(__name__)
infoLogger.setLevel(logging.INFO)
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

py_handler.setFormatter(py_formatter)
infoLogger.addHandler(py_handler)


class FalseParameters(Exception):
    # def __init__(self):
    #     super.__init__()
    pass


class SLAE:
    def __init__(self, r, n, k, start_variation=0, seed=0):
        self.r = r
        self.n = n
        self.k = k
        self.seed = seed
        self.start_v = start_variation
        self.param_check()

    def param_check(self) -> Any:
        """Проверка валидности параметров"""
        if self.r < 1 or self.n < 0 or self.k < 1:
            infoLogger.error(f"param({self.r, self.n, self.k}) < 0")
            raise FalseParameters(f"param({self.r, self.n, self.k}) < 0")
        if self.r - self.n < 1:
            infoLogger.error(f"r({self.r}) - n({self.n}) < 0")
            raise FalseParameters(f"r({self.r}) - n({self.n}) < 0")
        elif self.k < self.r - self.n:
            infoLogger.error(f"k({self.k}) < r({self.r}) - n({self.n})")
            raise FalseParameters(f"k({self.k}) < r({self.r}) - n({self.n})")
        else:
            infoLogger.info(f"k={self.k}, r={self.r}, n={self.n}")
        return True

    def generate_random_matrix(self) -> (np.ndarray, np.ndarray):
        rows = self.r - self.n
        cols = self.r
        k = self.k
        infoLogger.info(f"start_variant={self.start_v}, seed_value={self.seed}")
        rng = np.random.default_rng(self.seed)
        # пропускаем первые start_variant генераций
        _ = rng.random(self.start_v * rows * 2)

        matrix = rng.integers(-10, 11, size=(rows, cols))
        matrix[0][0] = 1
        b = rng.integers(-10, 11, size=(rows, 1))

        while np.linalg.matrix_rank(matrix) != rows:
            infoLogger.info(f"generation_false_attempt: matrix=\n{matrix}")
            matrix = rng.integers(-10, 11, size=(rows, cols))
            matrix[0][0] = 1
        infoLogger.info(f"generation_successful_attempt: matrix=\n{matrix}\nparameters b: \n{b}")

        for i in range(rows, k):
            coeffs = rng.integers(-5, 6, size=rows)
            # умножение matrix[j] * coeffs[j] с результатом вектором, например [1 3 0]*0 + [-5 -4 -10]*-4 = [20 5 50]
            matrix2 = np.dot(coeffs, matrix[:rows])
            matrix = np.append(matrix, [matrix2], axis=0)
            b2 = np.dot(coeffs, b)
            b = np.append(b, [b2], axis=0)
            infoLogger.info(f"new_row: row={matrix2} b={b2} coefficients={coeffs}")

        infoLogger.info(f"result_matrix: matrix=\n{matrix} \nb=\n{b}")
        return matrix, b

    def __str__(self):
        return "Heterogeneous SLAE"


class HomSLAE(SLAE):
    def generate_random_matrix(self) -> (np.ndarray, np.ndarray):
        matrix, b = super().generate_random_matrix()
        b = np.zeros(dtype=int, shape=(b.shape[0], b.shape[1]))
        return matrix, b

    def __str__(self):
        return "Homogeneous SLAE"


class SLAEParam(SLAE):
    def __init__(self, r, n, k, lmb, seed=0, start_variation=0):
        super().__init__(r, n, k, seed, start_variation)
        self.lmb = lmb

    def generate_random_matrix(self) -> (np.ndarray, np.ndarray):
        matrix, b = super().generate_random_matrix()
        b[self.k - 1][0] = self.lmb
        infoLogger.info(f"new b: \n{b}")
        return matrix, b

    def __str__(self):
        return "Heterogeneous parametric SLAE"


if __name__ == "__main__":
    our_SLAE: SLAE = SLAEParam(3, 1, 3, 0)
    print(our_SLAE.generate_random_matrix())