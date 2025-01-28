from math import prod


class Equation:
    name = None

    @staticmethod
    def parameters(X):
        raise NotImplementedError

    @staticmethod
    def default_parameters(X):
        raise NotImplementedError

    @staticmethod
    def calc(X, *P):
        # WARNING GeCo assumes that equations are commutative. If your equation is not, than
        # GeCo mitght not find the appropriate formula.
        raise NotImplementedError

    @staticmethod
    def print(X, P, precision=5):
        raise NotImplementedError


class SUM(Equation):
    name = "Sum"

    @staticmethod
    def parameters(X):
        return len(X) + 1

    @staticmethod
    def default_parameters(X):
        # First variable (A matrix) => 1.0
        # All other variables (B matrix) => 0.0
        # Error / offset => 0.0
        return [1.0] + [0.0] * (len(X) - 1) + [0.0]

    @staticmethod
    def calc(X, *P):
        assert SUM.parameters(X) == len(P)

        return sum([x * p for x, p in zip(X, P[:-1])]) + P[-1]

    @staticmethod
    def print(X, P, precision=5):
        s = ""
        for x, p in zip(X, P[:-1]):
            s += f"{x} * {round(p, precision)} + "
        s += f"{round(P[-1], precision)}"
        return s


class PRODUCT(Equation):
    name = "Product"

    @staticmethod
    def parameters(X):
        return 3

    @staticmethod
    def default_parameters(X):
        # First variable (A matrix) => 1.0
        # Second variable (B matrix) => 1.0
        # Error / offset => 0.0
        return [1.0, 1.0, 0.0]

    @staticmethod
    def calc(X, *P):
        assert len(X) >= 2
        assert PRODUCT.parameters(X) == len(P)
        return P[0] * X[0] + P[1] * prod(X[1:]) + P[2]

    @staticmethod
    def print(X, P, precision=5):
        s = f"{round(P[0], precision)} * {X[0]} "

        s += f"+ {round(P[1], precision)} "
        for x in X[1:]:
            s += f"* {x} "

        s += f"+ {round(P[2], precision)}"
        return s


equations = [SUM, PRODUCT]


def get_equation(name):
    for eq in equations:
        if eq.name == name:
            return eq
    raise Exception
