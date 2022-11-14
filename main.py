from time import sleep
from dual import Dual
from ml import *

# forward differentiation: O(N*inputs) # propagate from input # dual numbers
# backward differentiation: O(N*outputs) # propagate from output
# But you make (dx or dy) be a vector and take the derivative (with respect to/of) that.
# Here we just take one backward derivative with respect to all parameters which is just O(N)
# https://arxiv.org/pdf/1502.05767.pdf Table 3
# https://www.youtube.com/watch?v=R_m4kanPy6Q

def print_duals():
    x1 = Dual(2, 0)
    x2 = Dual(5, 0)
    print(f"{x1}; {x2}")
    v1 = x1.log()
    v2 = x1 * x2
    v3 = x2.sin()
    print(f"{v1}; {v2}; {v3}")
    v4 = v1 + v2
    v5 = v4 - v3
    print(f"{v4}; {v5}")

def print_backward_differentiation():
    a = NeuralNetwork()
    a.add_layer(LayerType.Input, 2)
    a.add_layer(LayerType.FullyConnected, 2).set_params([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    a.add_layer(LayerType.SquaredLoss, 2)
    #a.initialize()
    while True:
        cases = [
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            #([1.0, 1.0], [1.0, 0.0]),
        ]
        for i in range(1000):
            for input, output in cases:
                a.train(input, output)
            #print(a)
        print("; ".join(str(a.forward(input)) for input, output in cases))
        #print(a)
        #sleep(.1)

if __name__ == "__main__":
    #print_duals()
    print_backward_differentiation()
