from qLib.tests import *
from math import e
from dual import Dual
from ml import NeuralNetwork, LayerType

def assert_close(got: float, expected: float, epsilon=1e-9):
    if abs(got - expected) >= epsilon:
        raise AssertionError(f"got: {got}; expected: close {expected}")

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

@test
def learn_linear_xor():
    a = NeuralNetwork()
    a.add_layer(LayerType.Input, 2)
    a.add_layer(LayerType.FullyConnected, 2)
    a.add_layer(LayerType.SquaredLoss, 2)
    a.initialize()
    cases = [
        ([0.0, 0.0], [1.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0], [1.0, 0.0]),
    ]
    # 3 samples
    for i in range(50000):
        for input, output in cases[:3]:
            a.train(input, output)
    for input, output in cases[:3]:
        zero, one = a.forward(input)
        assert_close(zero, output[0])
        assert_close(one, output[1])
    # 4 samples (not possible with a linear function)
    a.initialize()
    for i in range(10000):
        for input, output in cases:
            a.train(input, output)
    for input, output in cases:
        zero, one = a.forward(input)
        assert_close(zero, 0.5, 1e-1)
        assert_close(one, 0.5, 1e-1)

# TODO: test_learn_xor

if __name__ == "__main__":
    run_tests()
