"""Message - посредник между разными типами SLAE и Encoder"""
import json
import numpy as np
from SLAE import SLAE
from surface_area import surface_area as SA
from sympy import Rational, pi, sqrt
import abc
import logging

import enum

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

    def __str__(self):
        return self.__doc__


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

    def __str__(self):
        return self.__doc__


class SurfaceAreaMsg(Message):
    """SurfaceAreaMsg - адаптер, который меняет своё поведение в зависимости от разных типов данных"""
    def __init__(self, s_a: SA.SurfaceArea):
        self.ans = s_a.generate_answer()
        self.s_a = s_a

    def encode(self, **kwargs) -> str:
        if isinstance(self.s_a, SA.SurfaceArea1):
            msg = dict()
            msg["1"] = dict()
            if Rational(self.s_a.a * 3).q != 1:
                frac_string = f"\\frac{{{Rational(self.s_a.a * 3).p}}}{{{Rational(self.s_a.a * 3).q}}}"
            else:
                if Rational(self.s_a.a * 3).p == 1:
                    frac_string = ""
                else:
                    frac_string = f"{self.s_a.a*3}"
            msg["1"]["condition"] = f" уравнением ${{{{y}}^{{2}}}}={{{self.s_a.a.q}}}\\frac{{x}}{{{9 * self.s_a.a.p}}}{{{{\\left( {frac_string}-x \\right)}}^{2}}}}}$с ограничениями $0 \\le x \\le 3$, ось вращения - $OX$"
            msg["1"]["answer"] = str(self.ans)
            a_rat = Rational(self.ans / pi)
            if a_rat.q == 1:
                msg["1"]["answerLatex"] = f"{a_rat}\\pi"
            else:
                msg["1"]["answerLatex"] = f"\\frac{{{a_rat.p}}}{{{a_rat.q}}}\\pi"
            if len(kwargs) != 0:
                return json.dumps(msg, ensure_ascii=False, **kwargs)
            return json.dumps(msg, ensure_ascii=False)
        if isinstance(self.s_a, SA.SurfaceArea2):
            msg = dict()
            msg["1"] = dict()
            if Rational(self.s_a.a * 3).q != 1:
                frac_string = f"\\frac{{{Rational(self.s_a.a * 3).p}}}{{{Rational(self.s_a.a * 3).q}}}"
            else:
                if Rational(self.s_a.a * 3).p == 1:
                    frac_string = ""
                else:
                    frac_string = f"{self.s_a.a * 3}"

            if self.s_a.a.q == 1:
                frac_string2 = ""
            else:
                frac_string2 = f"{{{self.s_a.a.q}}}"

            msg["1"][
                "condition"] = f" уравнением ${{{{y}}^{{2}}}}={frac_string2}\\frac{{x}}{{{9 * self.s_a.a.p}}}{{{{\\left( {frac_string}-x \\right)}}^{2}}}}}$с ограничениями $0 \\le x \\le 3$, ось вращения - $OY$"

            msg["1"]["answer"] = str(self.ans)

            a_rat = Rational(self.ans / pi / sqrt(3))
            if a_rat.q == 1:
                frac_string = f"{{{a_rat.p}}}"
            else:
                frac_string = f"\\frac{{{a_rat.p}}}{{{a_rat.q}}}"
            msg["1"]["answerLatex"] = f"{frac_string}\\pi\\sqrt{{3}}"
            if len(kwargs) != 0:
                return json.dumps(msg, ensure_ascii=False, **kwargs)
            return json.dumps(msg, ensure_ascii=False)
        if isinstance(self.s_a, SA.SurfaceArea3):
            msg = dict()
            msg["1"] = dict()
            if self.s_a.a.q != 1:
                frac_string = f"\\frac{{{self.s_a.a.p}}}{{{self.s_a.a.q}}}"
            else:
                if self.s_a.a.p == 1:
                    frac_string = ""
                else:
                    frac_string = f"{self.s_a.a}"

            if Rational(self.s_a.a / 3).q != 1:
                if Rational(self.s_a.a / 3).p == 1:
                    frac_string2 = f"\\frac{{t}}{{{Rational(self.s_a.a / 3).q}}}"
                else:
                    frac_string2 = f"\\frac{{{Rational(self.s_a.a / 3).p}t}}{{{Rational(self.s_a.a / 3).q}}}"
            else:
                if Rational(self.s_a.a / 3).p == 1:
                    frac_string2 = ""
                else:
                    frac_string2 = f"{{{Rational(self.s_a.a / 3).p}t}}"
            msg["1"][
                "condition"] = f" уравнением $\\begin{{cases}}  x={frac_string}\\left( {{{{t}}^{{2}}}}+1 \\right) \\\\ y={frac_string2}\\left( 3-{{{{t}}^{{2}}}} \\right)  \\end{{cases}}$ с ограничениями $1 \\le x \\le 4$, ось вращения - $OX$"

            msg["1"]["answer"] = str(self.ans)

            a_rat = Rational(self.ans / pi)
            if a_rat.q == 1:
                msg["1"]["answerLatex"] = f"{a_rat}\\pi"
            else:
                msg["1"]["answerLatex"] = f"\\frac{{{a_rat.p}}}{{{a_rat.q}}}\\pi"
            if len(kwargs) != 0:
                return json.dumps(msg, ensure_ascii=False, **kwargs)
            return json.dumps(msg, ensure_ascii=False)
        if isinstance(self.s_a, SA.SurfaceArea4):
            msg = dict()
            msg["1"] = dict()
            if self.s_a.a.q == 1:
                frac_string = ""
            else:
                frac_string = str(self.s_a.a.q)
            msg["1"][
                "condition"] = f" уравнением $r=\\frac{{{self.s_a.a.p}}}{{{frac_string}{{{{{{\\cos }}^{{2}}}}\\frac{{\\varphi }}{{2}}}}}}$ с ограничениями $0\\le \\varphi \\le \frac{{\\pi }}{{2}}$, ось вращения - $OX$"

            msg["1"]["answer"] = str(self.ans)

            a_rat = Rational(self.s_a.a * self.s_a.a * 8 / 3)
            if a_rat.q == 1:
                msg["1"]["answerLatex"] = f"{a_rat}\\pi{{2\\sqrt{{2}}-1}}"
            else:
                msg["1"]["answerLatex"] = f"\\frac{{{a_rat.p}}}{{{a_rat.q}}}\\pi\\left({{2\\sqrt{{2}}-1}}\\right)"
            if len(kwargs) != 0:
                return json.dumps(msg, ensure_ascii=False, **kwargs)
            return json.dumps(msg, ensure_ascii=False)


class Encoder:
    """Encoder - клиентский интерфейс, аггрегирует Message"""
    def __init__(self, msg: Message):
        self.msg = msg

    def json_encode(self, **kwargs) -> str:
        encoded_message = self.msg.encode(**kwargs)
        infoLogger.info("Message encoded")
        return encoded_message

    def __str__(self):
        return self.__doc__


class Generator:
    def __init__(self, obj, count=0):
        self.obj = obj
        self.count = count

    def gen(self):
        if isinstance(self.obj, SLAE.SLAE):
            for i in range(self.count):  # obj = SLAE.HomSLAE(...) / SLAE.SLAE(...)
                self.obj.set_start_var(i)
                slae_message: Message = SLAEMessage(self.obj)
                encoder = Encoder(slae_message)
                yield encoder.json_encode(indent=2)
        if isinstance(self.obj, SA.SurfaceArea): # obj = SA.SurfaceArea1(...) / SA.SurfaceArea2(...)
            if isinstance(self.obj, SA.SurfaceArea4):
                end = 18
            else:
                end = 27
            for i, sa in enumerate(self.obj):
               if i > end:
                   break
               surf_message: Message = SurfaceAreaMsg(sa)
               encoder = Encoder(surf_message)
               yield encoder.json_encode(indent=2)



        

if __name__ == "__main__":
    pass
    #slae = SLAE.HomSLAE(3, 0, 3)
    #slae_message: Message = SLAEMessage(slae)
    #encoder = Encoder(slae_message)
    #print(encoder.json_encode(indent=2))
    #
    #slae = SLAE.SLAEParam(3, 0, 3, 1)
    #slae_message: Message = SLAEMessage(slae)
    #encoder = Encoder(slae_message)
    #print(encoder.json_encode(indent=2))

    #surf1: SA.SurfaceArea = SA.SurfaceArea1(3, 3)
    #surf_message: Message = SurfaceAreaMsg(surf1)
    #encoder = Encoder(surf_message)
    #print(encoder.json_encode(indent=2))

    #for i, sa in enumerate(surf1):
    #    if i > 27:
    #        break
    #    surf_message: Message = SurfaceAreaMsg(sa)
    #    encoder = Encoder(surf_message)
    #    print(encoder.json_encode(indent=2))

    #surf2: SA.SurfaceArea = SA.SurfaceArea2(3, 3)
    #surf_message: Message = SurfaceAreaMsg(surf2)
    #encoder = Encoder(surf_message)
    #print(encoder.json_encode(indent=2))

    # for i, sa in enumerate(surf2):
    #    if i > 27:
    #        break
    #    surf_message: Message = SurfaceAreaMsg(sa)
    #    encoder = Encoder(surf_message)
    #    print(encoder.json_encode(indent=2))

    #surf3: SA.SurfaceArea = SA.SurfaceArea3(3, 3)
    #surf_message: Message = SurfaceAreaMsg(surf3)
    #encoder = Encoder(surf_message)
    #print(encoder.json_encode(indent=2))

    #for i, sa in enumerate(surf3):
    #   if i > 27:
    #       break
    #   surf_message: Message = SurfaceAreaMsg(sa)
    #   encoder = Encoder(surf_message)
    #   print(encoder.json_encode(indent=2))
    #
    #surf4: SA.SurfaceArea = SA.SurfaceArea4(2, 2)
    #surf_message: Message = SurfaceAreaMsg(surf4)
    #encoder = Encoder(surf_message)
    #print(encoder.json_encode(indent=2))
    #
    #for i, sa in enumerate(surf4):
    #   if i > 27:
    #       break
    #   surf_message: Message = SurfaceAreaMsg(sa)
    #   encoder = Encoder(surf_message)
    #   print(encoder.json_encode(indent=2))