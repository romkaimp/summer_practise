import unittest
import logging
from SLAE import homogeneous_SLAE as SLAE1
import numpy as np

infoLogger = logging.getLogger(__name__)
infoLogger.setLevel(logging.INFO)
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
infoLogger.addHandler(py_handler)


def generate_params() -> np.ndarray:
    return np.random.randint(0, 10, 3)


class SLAE1Tests(unittest.TestCase):
    def test_validity(self):
        params = [generate_params() for i in range(100)]
        for i, rnk in enumerate(params):
            infoLogger.info(f"test_validity {i}-iteration {rnk[0], rnk[1], rnk[2]}-params")
            try:
                matrix = SLAE1.generate_random_matrix(rnk[0]-rnk[1], rnk[0], rnk[2], 0, 0)
                self.assertIsInstance(matrix, np.ndarray, msg="isInstance")
                self.assertEqual(matrix.shape[0], rnk[2])  # количество строк в матрице = k
                self.assertEqual(matrix.shape[1], rnk[0])  # количество столбцов в матрице = r
            except SLAE1.FalseParameters as err:
                self.assertRaises(SLAE1.FalseParameters, SLAE1.param_check, *rnk)

    def test_matrix_rank(self):
        params = [generate_params() for i in range(100)]
        for i, rnk in enumerate(params):
            infoLogger.info(f"test_matrix_rank {i}-iteration {rnk[0], rnk[1], rnk[2]}-params")
            try:
                matrix = SLAE1.generate_random_matrix(rnk[0] - rnk[1], rnk[0], rnk[2], 0, 0)
                self.assertEqual(np.linalg.matrix_rank(matrix), rnk[0] - rnk[1])
                self.assertEqual(matrix.shape[0], rnk[2])  # количество строк в матрице = k
                self.assertEqual(matrix.shape[1], rnk[0])  # количество столбцов в матрице = r
            except SLAE1.FalseParameters as err:
                self.assertRaises(SLAE1.FalseParameters, SLAE1.param_check, *rnk)

