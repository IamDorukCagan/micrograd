# micrograd

This is a version of the Micrograd library built from scratch. Its autograd engine can be extended to support tensor operations.

## What's here

| File | What it is |
| --- | --- |
| [`micrograd/engine.py`](micrograd/engine.py) | `Value`: a scalar that records the operations used to build it and backpropagates gradients through them (`+`, `*`, `**`, `/`, `-`, `tanh`, `exp`) |
| [`micrograd/nn.py`](micrograd/nn.py) | `Neuron`, `Layer` and `MLP` built on top of `Value` |
| [`micrograd/viz.py`](micrograd/viz.py) | `draw_dot`: draws the computation graph with graphviz, showing each node's data and gradient |
| [`micrograd.ipynb`](micrograd.ipynb) | `Value`, `draw_dot`, `Neuron`, `Layer` and `MLP` all together in one notebook |
| [`example.ipynb`](example.ipynb) | Examples using these structures and functions |

## Example

```python
from micrograd import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])

for k in range(100):
    ypred = [n(x) for x in xs]
    loss = sum((yp - y)**2 for y, yp in zip(ys, ypred))
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data += -0.01 * p.grad
```

The loss goes from about 4.7 to about 0.05 in 100 steps.

## Setup

```bash
python3.12 -m venv .venv   # PyTorch has no Python 3.14 builds yet; 3.12 works
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook example.ipynb
```

`draw_dot` also needs the Graphviz system package (`brew install graphviz` on macOS).
