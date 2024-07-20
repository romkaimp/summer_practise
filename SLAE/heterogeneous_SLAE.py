"""неоднородная СЛАУ"""
import logging

import numpy as np


class FalseParameters(Exception):
    # def __init__(self):
    #     super.__init__()
    pass

infoLogger = logging.getLogger(__name__)
infoLogger.setLevel(logging.INFO)
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

py_handler.setFormatter(py_formatter)
infoLogger.addHandler(py_handler)


def param_check(r, n, k) -> bool:
    """Проверка валидности параметров"""
    if r < 1 or n < 0 or k < 1:
        infoLogger.error(f"param({r, n, k}) < 0")
        raise FalseParameters(f"param({r, n, k}) < 0")
    if r - n < 1:
        infoLogger.error(f"r({r}) - n({n}) < 0")
        raise FalseParameters(f"r({r}) - n({n}) < 0")
    elif k < r - n:
        infoLogger.error(f"k({k}) < r({r}) - n({n})")
        raise FalseParameters(f"k({k}) < r({r}) - n({n})")
    else:
        infoLogger.info(f"k={k}, r={r}, n={n}")
    return True


seed_value = 0
start_variant = 0


def generate_random_matrix(rows, cols, k, seed=seed_value, start_variant=start_variant) -> (np.ndarray, np.ndarray):
    infoLogger.info(f"start_variant={start_variant}, seed_value={seed_value}")
    param_check(cols, cols-rows, k) # param_check(r, n, k)
    rng = np.random.default_rng(seed)
    # пропускаем первые start_variant генераций
    _ = rng.random(start_variant * rows * 2)

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

if __name__ == "__main__":
    k = 3  # int(input())
    r = 3  # intinput())
    n = 1  # int(input())
    print(generate_random_matrix(r - n, r, k))