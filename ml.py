from random import random
from typing import Union, cast
from qLib import assert_equals, Enum

class LayerType(Enum):
    Input = 0
    FullyConnected = 1
    SquaredLoss = 2
    LeakyReLU = 3

LEAKY_ReLU_K = 0.1

class _Layer:
    prev_layer: Union["_Layer", None]
    type: int
    values: list[float]
    value_adjoints: list[float] # dy/dv
    params: list[float]

    def __init__(self, prev_layer: Union["_Layer", None], type: int, size: int):
        self.prev_layer = prev_layer
        self.type = type
        self.values = [0.0] * (size+1)
        self.values[-1] = 1.0
        self.value_adjoints = [0.0] * size
        self.params = []
        if type == LayerType.FullyConnected:
            self.params = [0.0] * size * len(cast(_Layer, self.prev_layer).values)
        elif type == LayerType.SquaredLoss:
            self.params = [0.0] * size
            assert_equals(size + 1, len(cast(_Layer, self.prev_layer).values))
        elif type == LayerType.LeakyReLU:
            assert_equals(size + 1, len(cast(_Layer, self.prev_layer).values))

    def __repr__(self):
        def print_floats(floats: list[float]):
            return "[" + ", ".join(f"{v:.2f}" for v in floats) + "]"

        return f"(type={LayerType.toString(self.type)}, values={(self.values)}, params={print_floats(self.params)}, value_adjoints={print_floats(self.value_adjoints)}, param_adjoints={print_floats(self.get_param_adjoints())})"

    def set_values(self, values: list[float]):
        assert_equals(len(values), len(self.values) - 1)
        for i, v in enumerate(values):
            self.values[i] = float(v)

    def set_params(self, params: list[float]):
        assert_equals(len(params), len(self.params))
        self.params = [float(v) for v in params]

    def _reset_adjoints(self):
        self.value_adjoints = [0.0] * len(self.value_adjoints)

    # dy/dp
    def get_param_adjoints(self):
        if self.type == LayerType.FullyConnected:
            acc = [0.0] * len(self.params)
            n = len(self.values) - 1
            prev_layer = cast(_Layer, self.prev_layer)
            prev_n = len(prev_layer.values)
            for i in range(n):
                for j in range(prev_n):
                    acc[i + j*n] += self.value_adjoints[i] * prev_layer.values[j]
            return acc
        return []

    def forward(self):
        n = len(self.values) - 1
        if self.type == LayerType.FullyConnected:
            prev_layer = cast(_Layer, self.prev_layer)
            prev_n = len(prev_layer.values)
            self.set_values([0.0] * n)
            for i in range(n):
                for j in range(prev_n):
                    self.values[i] += self.params[i + j*n] * prev_layer.values[j]
        elif self.type == LayerType.SquaredLoss:
            prev_layer = cast(_Layer, self.prev_layer)
            self.set_values(prev_layer.values[:-1])
        elif self.type == LayerType.LeakyReLU:
            prev_layer = cast(_Layer, self.prev_layer)
            for j in range(len(self.values) - 1):
                prev_v = prev_layer.values[j]
                self.values[j] = prev_v if (prev_v > 0) else prev_v * LEAKY_ReLU_K

    def backward(self):
        n = len(self.values) - 1
        if self.type == LayerType.FullyConnected:
            prev_layer = cast(_Layer, self.prev_layer)
            prev_n = len(prev_layer.value_adjoints)
            prev_layer._reset_adjoints()
            for j in range(prev_n):
                for i in range(n):
                    prev_layer.value_adjoints[j] += self.params[i + j*n] * self.value_adjoints[i]
        elif self.type == LayerType.SquaredLoss:
            prev_layer = cast(_Layer, self.prev_layer)
            prev_layer._reset_adjoints()
            for i in range(n):
                prev_layer.value_adjoints[i] += 2 * (self.values[i] - self.params[i]) * self.value_adjoints[i]
        elif self.type == LayerType.LeakyReLU:
            prev_layer = cast(_Layer, self.prev_layer)
            prev_layer._reset_adjoints()
            for i in range(n):
                prev_layer.value_adjoints[i] = (1 if (self.values[i] > 0) else LEAKY_ReLU_K) * self.value_adjoints[i]

class NeuralNetwork:
    layers: list[_Layer]

    def __init__(self):
        self.layers = []

    def __repr__(self):
        NEWLINE = "\n"
        return f"[{''.join(f'{NEWLINE}  {v}' for v in self.layers)}\n]"

    def add_layer(self, type: int, size: int):
        layer = _Layer(self.layers[-1] if self.layers else None, type, size)
        self.layers.append(layer)
        return layer

    def forward(self, input: list[float]):
        self.layers[0].set_values(input)
        for layer in self.layers[1:]:
            layer.forward()
        return self.layers[-1].values[:-1]

    def backward(self, output: list[float]):
        self.layers[-1].value_adjoints = [1.0] * len(self.layers[-1].values)
        self.layers[-1].set_params(output)
        for layer in self.layers[-1:0:-1]:
            layer.backward()

    def initialize(self):
        for layer in self.layers:
            for i in range(len(layer.params)):
                layer.params[i] = 2 * random() - 1

    def train(self, input: list[float], output: list[float]):
        self.forward(input)
        self.backward(output)
        for layer in self.layers:
            for i, dp in enumerate(layer.get_param_adjoints()):
                layer.params[i] -= dp * 1e-3 # TODO: divide by network.parameter_count
