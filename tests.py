from qLib.tests import *
from math import e
from dual import Dual
from ml import NeuralNetwork, LayerType

def assert_close(got: float, expected: float, epsilon=1e-9, suffix=""):
    if abs(got - expected) > epsilon:
        raise AssertionError(f"got: {got}; expected: close {expected}{f'; suffix: {suffix}' if suffix else ''}")

def assert_not_close(got: float, expected: float, epsilon=1e-9, suffix=""):
    if abs(got - expected) <= epsilon:
        raise AssertionError(f"got: {got}; expected: close {expected}{f'; suffix: {suffix}' if suffix else ''}")

def check_function(f: Callable[[Dual], Dual], g: Callable[[float], float]):
    for x in [1, 2, 3]:
        assert_close(f(Dual(x, 1)).dx, g(x))

@test
def forward_linear_function():
    check_function(lambda x: x * Dual(2), lambda x: 2)

@test
def forward_quadratic_function():
    check_function(lambda x: x * x, lambda x: 2 * x)

@test
def forward_reciprocal_function():
    check_function(lambda x: Dual(1) / x, lambda x: -(x**-2))

@test
def forward_exponential_function():
    check_function(lambda x: x.exp(), lambda x: e**x)

XOR_CASES = [
    ([0.0, 0.0], [1.0, 0.0]),
    ([0.0, 1.0], [0.0, 1.0]),
    ([1.0, 0.0], [0.0, 1.0]),
    ([1.0, 1.0], [1.0, 0.0]),
]

@test
def learn_linear_xor():
    a = NeuralNetwork()
    a.add_layer(LayerType.Input, 2)
    a.add_layer(LayerType.FullyConnected, 2)
    a.add_layer(LayerType.SquaredLoss)
    a.initialize()
    # 3 samples
    for i in range(10000):
        for input, output in XOR_CASES[:3]:
            a.train(input, output)
    for input, output in XOR_CASES[:3]:
        zero, one = a.forward(input)
        assert_close(zero, output[0])
        assert_close(one, output[1])
    # 4 samples (not possible with a linear function)
    a.initialize()
    for i in range(10000):
        for input, output in XOR_CASES:
            a.train(input, output)
    for input, output in XOR_CASES:
        zero, one = a.forward(input)
        assert_not_close(zero, output[0], 1e-1)
        assert_not_close(one, output[1], 1e-1)

@test
def learn_xor():
    # TODO: take best of N?
    a = NeuralNetwork()
    a.add_layer(LayerType.Input, 2)
    a.add_layer(LayerType.FullyConnected, 2)
    a.add_layer(LayerType.LeakyReLU)
    a.add_layer(LayerType.FullyConnected, 2)
    a.add_layer(LayerType.SquaredLoss)
    # 4 samples
    a.initialize()
    for i in range(10000):
        for input, output in XOR_CASES:
            a.train(input, output)
    error_suffix = ""
    for input, output in XOR_CASES:
        zero, one = a.forward(input)
        error_suffix += f"\n{input} {[zero, one]}"
    for input, output in XOR_CASES:
        zero, one = a.forward(input)
        assert_close(zero, output[0], 1e-1, suffix=error_suffix)
        assert_close(one, output[1], 1e-1, suffix=error_suffix)

if __name__ == "__main__":
    run_tests()
