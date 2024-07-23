import numpy as np

import logging
import math
from typing import Any
from sympy import Rational, pi, sqrt


infoLogger = logging.getLogger(__name__)
infoLogger.setLevel(logging.INFO)
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

py_handler.setFormatter(py_formatter)
infoLogger.addHandler(py_handler)


class SurfaceArea:
    def __init__(self, a, b):
        self.a = Rational(a, b)
        self.var = 1

    def set_var(self, var):
        self.var = var

    def generate_answer(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        b = self.__class__(self.a.p, self.a.q)
        b.set_var(self.var)
        self.a += Rational(1, 3)
        self.var += 1
        return b


class SurfaceArea1(SurfaceArea):
    def generate_answer(self) -> float:
        return Rational(3 * self.a * self.a) * pi


class SurfaceArea2(SurfaceArea):
    def generate_answer(self) -> float:
        return Rational(56 * self.a * self.a, 5) * pi * sqrt(3)


class SurfaceArea3(SurfaceArea):
    def generate_answer(self) -> float:
        return Rational(3 * self.a * self.a) * pi


class SurfaceArea4(SurfaceArea):
    def generate_answer(self) -> float:
        return Rational(8 * self.a * self.a, 3) * pi * (2 * sqrt(2) - 1)

    def __next__(self):
        b = self.__class__(self.a.p, self.a.q)
        b.set_var(self.var)
        self.a += Rational(1, 2)
        self.var += 1
        return b


if __name__ == "__main__":
    surf1: SurfaceArea = SurfaceArea1(4, 3)
    print(surf1.generate_answer())

    #surf2: SurfaceArea = SurfaceArea2(3, 3)
    #print(surf2.generate_answer())
    #
    #for i, el in enumerate(surf1):
    #    if i > 27:
    #        break
    #    print(i, el.generate_answer())
    #
    #for i, el in enumerate(surf2):
    #    if i > 27:
    #        break
    #    print(i, el.generate_answer())

    print(Rational(surf1.generate_answer()/pi/3))