import json

import numpy
import numpy as np

import logging
from SLAE import homogeneous_SLAE as hom_SLAE, heterogeneous_SLAE as het_SLAE, heterogeneous_SLAE_param as het_SLAE_p


import abc

infoLogger = logging.getLogger(__name__)
infoLogger.setLevel(logging.INFO)
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
infoLogger.addHandler(py_handler)


class Message(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self) -> str:
        pass


class HomogeneousSLAE(Message):
    def __init__(self, r, n, k, seed, start_value):
        """r - количество неизвестных, n - количество независымых решений, к - количество уравнений"""
        self.r = r
        self.n = n
        self.k = k
        self.matrix: np.ndarray = hom_SLAE.generate_random_matrix(r-n, r, k, seed, start_value)

    def _get_raw(self) -> numpy.ndarray:
        for i in self.matrix:
            yield i

    @staticmethod
    def _get_eq(raw) -> str:
        res = []
        res.append(f"{raw[0]}x_1")
        for i in range(len(raw)-1):
            if raw[i+1] > 0:
                res.append(f"+{raw[i+1]}x_{i+2}")
            else:
                res.append(f"{raw[i + 1]}x_{i + 2}")
        res.append("=0")
        return ("").join(res)

    def _get_eqs(self):
        res = []
        gen = self._get_raw()
        for raw in gen:
            eq = self._get_eq(raw)
            res.append(eq)
        return ("\\\\").join(res)

    def encode(self, **kwargs) -> str:
        msg = dict()
        msg["1"] = dict()
        msg["1"]["condition"] = self._get_eqs()
        msg["1"]["isSolution"] = "yes"
        msg["1"]["rang"] = self.r - self.n
        msg["1"]["n"] = self.n
        msg["1"]["A"] = self.matrix.tolist()
        msg["1"]["answerLatex"] = "\\\\operatorname{rang} \\left( A \\left| B \\right. \\right)"+f" = {self.r-self.n}"
        if len(kwargs) != 0:
            return json.dumps(msg, **kwargs)
        return json.dumps(msg)


class HeterogeneousSLAE(Message):
    def __init__(self, r, n, k, seed, start_value):
        """r - количество неизвестных, n - количество независымых решений, к - количество уравнений"""
        self.r = r
        self.n = n
        self.k = k
        self.matrix: np.ndarray
        self.b: np.ndarray
        self.matrix, self.b = het_SLAE.generate_random_matrix(r - n, r, k, seed, start_value)

    def _get_raw(self) -> numpy.ndarray:
        for i, j in zip(self.matrix, self.b):
            yield i, j


    @staticmethod
    def _get_eq(raw, b_p) -> str:
        res = []
        res.append(f"{raw[0]}x_1")
        for i in range(len(raw) - 1):
            if raw[i + 1] >= 0:
                res.append(f"+{raw[i + 1]}x_{i + 2}")
            else:
                res.append(f"{raw[i + 1]}x_{i + 2}")

        res.append(f"={b_p[0]}")
        return ("").join(res)

    def _get_eqs(self):
        res = []
        gen = self._get_raw()
        for raw, b_p in gen:
            eq = self._get_eq(raw, b_p)
            res.append(eq)
        return ("\\\\").join(res)

    def encode(self, **kwargs) -> str:
        msg = dict()
        msg["1"] = dict()
        msg["1"]["condition"] = self._get_eqs()
        msg["1"]["answer"] = dict()
        msg["1"]["answer"]["isConsistent"] = "yes"
        msg["1"]["answer"]["rang"] = self.r - self.n
        msg["1"]["answer"]["n"] = self.n
        msg["1"]["answer"]["A"] = self.matrix.tolist()
        msg["1"]["answer"]["b"] = self.b.reshape(self.k).tolist()
        #msg["1"]["answerLatex"] = "\\\\operatorname{rang} \\left( A \\left| B \\right. \\right)" + f" = {self.r - self.n}"
        if len(kwargs) != 0:
            return json.dumps(msg, **kwargs)
        return json.dumps(msg)


class HeterogeneousSLAEParametric(Message):
    def __init__(self, r, n, k, lmb, seed, start_value):
        """r - количество неизвестных, n - количество независымых решений, к - количество уравнений, lmb - параметр"""
        self.r = r
        self.n = n
        self.k = k
        self.lmb = lmb
        self.matrix: np.ndarray
        self.b: np.ndarray
        self.matrix, self.b = het_SLAE_p.generate_random_matrix(r - n, r, k, lmb, seed, start_value)

    def _get_raw(self) -> numpy.ndarray:
        for i, j in zip(self.matrix, self.b):
            yield i, j


    @staticmethod
    def _get_eq(raw, b_p) -> str:
        res = []
        res.append(f"{raw[0]}x_1")
        for i in range(len(raw) - 1):
            if raw[i + 1] >= 0:
                res.append(f"+{raw[i + 1]}x_{i + 2}")
            else:
                res.append(f"{raw[i + 1]}x_{i + 2}")

        res.append(f"={b_p[0]}")
        return ("").join(res)

    def _get_eqs(self):
        res = []
        gen = self._get_raw()
        for raw, b_p in gen:
            eq = self._get_eq(raw, b_p)
            res.append(eq)
        return ("\\\\").join(res)

    def encode(self, **kwargs) -> str:
        msg = dict()
        msg["1"] = dict()
        msg["1"]["condition"] = self._get_eqs()
        msg["1"]["answer"] = dict()
        #msg["1"]["answer"]["isConsistent"] = "yes"
        #msg["1"]["answer"]["rang"] = self.r - self.n
        msg["1"]["answer"]["lambda"] = self.lmb
        msg["1"]["answer"]["n"] = self.n
        msg["1"]["answer"]["A"] = self.matrix.tolist()
        msg["1"]["answer"]["b"] = self.b.reshape(self.k).tolist()
        #msg["1"]["answerLatex"] = "\\\\operatorname{rang} \\left( A \\left| B \\right. \\right)" + f" = {self.r - self.n}"
        msg["1"]["answerLatex"] = f"\\lambda = {self.lmb}"
        if len(kwargs) != 0:
            return json.dumps(msg, **kwargs)
        return json.dumps(msg)


class Encoder:
    def __init__(self, msg: Message):
        self.msg = msg

    def json_encode(self, **kwargs):
        return self.msg.encode(**kwargs)

if __name__ == "__main__":
    r, n, k, seed, start_value = 3, 1, 3, 1, 0  # rang=2

    # new_SLAE: Message = HomogeneousSLAE(r, n, k, seed, start_value)
    # SLAE_Encoder = Encoder(new_SLAE)
    # print(SLAE_Encoder.json_encode(indent=2))

    # new_SLAE_het: Message = HeterogeneousSLAE(r, n, k, seed, start_value)
    # SLAE_Encoder = Encoder(new_SLAE_het)
    # print(SLAE_Encoder.json_encode(indent=2))

    lmb = 7
    new_SLAE_het_p: Message = HeterogeneousSLAEParametric(r, n, k, lmb, seed, start_value)
    SLAE_Encoder = Encoder(new_SLAE_het_p)
    print(SLAE_Encoder.json_encode(indent=2))
