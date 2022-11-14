# ml
PoC machine learning

```py
a = NeuralNetwork()
a.add_layer(LayerType.Input, 2)
a.add_layer(LayerType.FullyConnected, 2)
a.add_layer(LayerType.SquaredLoss, 2)
a.initialize()
cases = [
    ([0.0, 0.0], [1.0, 0.0]),
    ([0.0, 1.0], [0.0, 1.0]),
    ([1.0, 0.0], [0.0, 1.0]),
    #([1.0, 1.0], [1.0, 0.0]),
]
for i in range(10000):
    for input, output in cases:
        a.train(input, output)
for input, output in cases:
    print(a.forward(input))
    # [0.9959223329329098, 0.0022968551626328057]
    # [0.0014833122608711724, 0.999164485129832]
    # [0.001487387522970729, 0.9991621886722373]
```
