"""Message - посредник между разными типами SLAE и Encoder"""
import json
import numpy as np
from SLAE import SLAE
import abc
import logging

infoLogger = logging.getLogger(__name__)
infoLogger.setLevel(logging.INFO)
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
py_handler.setFormatter(py_formatter)
infoLogger.addHandler(py_handler)


class Message(metaclass=abc.ABCMeta):
    """Message - абстрактный класс, который инкапсулирует методы для работы с разными данными"""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self, **kwargs) -> str:
        pass


class SLAEMessage(Message):
    """SLAEMessage - адаптер, который меняет своё поведение в зависимости от разных типов данных"""
    def __init__(self, sl: SLAE):
        self.SLAE: SLAE = sl
        self.matrix: np.ndarray
        self.b: np.ndarray
        self.matrix, self.b = self.SLAE.generate_random_matrix()
        infoLogger.info("New message")

    def _get_raw(self) -> np.ndarray:
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
        infoLogger.info(f"SLAE type: {self.SLAE}")
        if isinstance(self.SLAE, SLAE.SLAEParam):
            msg = dict()
            msg["1"] = dict()
            msg["1"]["condition"] = self._get_eqs()
            msg["1"]["answer"] = dict()
            # msg["1"]["answer"]["isConsistent"] = "yes"
            # msg["1"]["answer"]["rang"] = self.r - self.n
            msg["1"]["answer"]["lambda"] = self.SLAE.lmb
            msg["1"]["answer"]["n"] = self.SLAE.n
            msg["1"]["answer"]["A"] = self.matrix.tolist()
            msg["1"]["answer"]["b"] = self.b.reshape(self.SLAE.k).tolist()
            # msg["1"]["answerLatex"] = "\\\\operatorname{rang} \\left( A \\left| B \\right. \\right)" + f" = {self.r - self.n}"
            msg["1"]["answerLatex"] = f"\\lambda = {self.SLAE.lmb}"
            if len(kwargs) != 0:
                return json.dumps(msg, **kwargs)
            return json.dumps(msg)
        elif isinstance(self.SLAE, SLAE.HomSLAE):
            msg = dict()
            msg["1"] = dict()
            msg["1"]["condition"] = self._get_eqs()
            msg["1"]["isSolution"] = "yes"
            msg["1"]["rang"] = self.SLAE.r - self.SLAE.n
            msg["1"]["n"] = self.SLAE.n
            msg["1"]["A"] = self.matrix.tolist()
            msg["1"][
                "answerLatex"] = "\\\\operatorname{rang} \\left( A \\left| B \\right. \\right)" + f" = {self.SLAE.r - self.SLAE.n}"
            if len(kwargs) != 0:
                return json.dumps(msg, **kwargs)
            return json.dumps(msg)
        elif isinstance(self.SLAE, SLAE.SLAE):
            msg = dict()
            msg["1"] = dict()
            msg["1"]["condition"] = self._get_eqs()
            msg["1"]["answer"] = dict()
            msg["1"]["answer"]["isConsistent"] = "yes"
            msg["1"]["answer"]["rang"] = self.SLAE.r - self.SLAE.n
            msg["1"]["answer"]["n"] = self.SLAE.n
            msg["1"]["answer"]["A"] = self.matrix.tolist()
            msg["1"]["answer"]["b"] = self.b.reshape(self.SLAE.k).tolist()
            # msg["1"]["answerLatex"] = "\\\\operatorname{rang} \\left( A \\left| B \\right. \\right)" + f" = {self.r - self.n}"
            if len(kwargs) != 0:
                return json.dumps(msg, **kwargs)
            return json.dumps(msg)


class Encoder:
    """Encoder - клиентский интерфейс, аггрегирует Message"""
    def __init__(self, msg: Message):
        self.msg = msg

    def json_encode(self, **kwargs) -> str:
        encoded_message = self.msg.encode(**kwargs)
        infoLogger.info("Message encoded")
        return encoded_message


if __name__ == "__main__":
    slae = SLAE.HomSLAE(3, 0, 3)
    slae_message: Message = SLAEMessage(slae)
    encoder = Encoder(slae_message)
    print(encoder.json_encode(indent=2))

    slae = SLAE.SLAEParam(3, 0, 3, 1)
    slae_message: Message = SLAEMessage(slae)
    encoder = Encoder(slae_message)
    print(encoder.json_encode(indent=2))