import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range (nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout): ## nin -> basically number of weights, nout = number of neuorons in the layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x): ##nin -> number of
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts #to initialize with the input
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #create layers while iterating, the o/p of the first layer becomes the i/p of the next
        #that is Layer(nin, nout1), Layer(nout1, nin) -> what happens here is now the input dimension is not x dimension but it matches the num of outputs of 1st layer
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) #in every iteration the output becomes the input and fed into the next layer which have correct dimension to take this input
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
