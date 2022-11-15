from time import sleep
from dual import Dual
from ml import *

# forward differentiation:
# - propagate from input
# - compute dv/dx for every v
# - implemented using dual numbers
# - O(N*inputs)
# backward differentiation:
# - propagate from output
# - compute dy/dv for every v
# - implemented using chain rule starting with dy/dy = 1
#   We can compute dy/dp as (dy/dv) * (dv/dp) where dv/dp is (f(p+dp)-f(p))/dp,
#   for example:
#    a) f(v) = v+w
#       df/dv = (v+dv+w - (v+w))/dv = dv/dv = 1
#    b) f(v) = v*p
#       df/dv = ((v+dv)*p-v*p)/dv = p*dp/dp = p
#    c) f(p) = v*p
#       df/dp = (v*(p+dp)-v*p)/dv = v*dp/dp = v
#    d) L(v) = (v-Ev)^2
#       dL/dv = ((v+dv-Ev)^2 - (v-Ev)^2) / dv = (dv^2 + 2(v - Ev)dv)/dv = dv + 2(v - Ev) = 2(v - Ev)
#    e) LeakyReLU(v) = v if (v > 0) else k*v
#       dLeakyReLU/dv = (f(v+dv) - f(v)) / dv =
#        v > 0: dv/dv = 1
#        v <= 0: kdv/dv = k
# - O(N*outputs)
#   You can make (dx or dy) be a vector and take the derivative (with respect to/of) that.
#   Here we take one backward derivative with (dy = y) with respect to all parameters, which is just O(N).

def print_forward_differentiation():
    x1 = Dual(2, 1) # set nonzero dx for the direction in which to take the derivative
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
    a.add_layer(LayerType.FullyConnected, 2)
    a.add_layer(LayerType.LeakyReLU)
    a.add_layer(LayerType.FullyConnected, 2)
    a.add_layer(LayerType.SquaredLoss)
    a.initialize()
    i = 0
    while True:
        cases = [
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [1.0, 0.0]),
        ]
        for j in range(1000):
            for input, output in cases:
                a.train(input, output)
                i += 1
            #print(a)
        print(f"{i} " + "; ".join(str(a.forward(input)) for input, output in cases))
        #print(a)
        #sleep(.1)

if __name__ == "__main__":
    #print_forward_differentiation()
    print_backward_differentiation()
