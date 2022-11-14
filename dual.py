from math import e, sin, cos, log

# dual numbers from geometric algebra
class Dual:
    x: float
    dx: float

    def __init__(self, x, dx=0.0):
        self.x = float(x)
        self.dx = float(dx)

    def __repr__(self):
        return f"(x={self.x}, dx={self.dx})"

    def __add__(self, other):
        return Dual(self.x + other.x, self.dx + other.dx)

    def __sub__(self, other):
        return Dual(self.x - other.x, self.dx - other.dx)

    def __mul__(self, other):
        return Dual(self.x * other.x, self.x * other.dx + self.dx * other.x)

    def __truediv__(self, other):
        return self * other.multiplicative_inverse()

    def multiplicative_inverse(self):
        inv_norm_squared = 1 / (self.x * self.x)
        return Dual(self.x * inv_norm_squared, -self.dx * inv_norm_squared)

    def exp(self):
        return Dual(e**self.x) * Dual(1, self.dx)

    def log(self):
        return Dual(log(self.x), self.dx / self.x)

    def sin(self):
        return Dual(sin(self.x), cos(self.x) * self.dx)

    def cos(self):
        return Dual(cos(self.x), -sin(self.x) * self.dx)
