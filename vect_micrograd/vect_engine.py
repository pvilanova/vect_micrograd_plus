"""A vectorized NumPy autograd engine for micrograd.

This keeps the micrograd idea (dynamic DAG + reverse-mode autodiff), but each
Value stores a whole NumPy array instead of one Python scalar. That changes the
cost of a dense layer from thousands of tiny scalar nodes to a handful of array
ops: matmul, add, ReLU, sum/mean.
"""

from __future__ import annotations

import numpy as np


def _as_array(data):
    """Convert numbers/lists/arrays to a float ndarray."""
    return np.array(data, dtype=float)

def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcasted gradient back down to the original operand shape.

    Example: if b has shape (32,) and out = X + b has shape (200, 32), then
    dL/db is the row-sum of dL/dout, not the full (200, 32) array.
    """
    grad = np.asarray(grad, dtype=float)

    # Scalars receive the sum of all gradient contributions.
    if shape == ():
        return np.array(grad.sum(), dtype=float)

    # Remove leading dimensions that were added by NumPy broadcasting.
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # Dimensions of size 1 were stretched; sum them back with keepdims=True.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Value:
    """Stores a NumPy array and its gradient."""

    __array_priority__ = 1000

    def __init__(self, data, _children=(), _op: str = ""):
        self.data = _as_array(data)
        self.grad = np.zeros_like(self.data, dtype=float)

        # Internal variables used for autograd graph construction.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Matrix multiplication
    # -------------------------------------------------------------------------
    #
    # This implementation intentionally supports only the cases needed for this
    # educational vectorized micrograd:
    #
    #   1. Matrix @ Matrix
    #        (batch, in_features) @ (in_features, out_features)
    #        -> (batch, out_features)
    #
    #      This is the standard dense-layer case:
    #
    #        y = x @ W
    #
    #      Backward:
    #
    #        dL/dx = dL/dy @ W.T
    #        dL/dW = x.T @ dL/dy
    #
    #
    #   2. Vector @ Matrix
    #        (in_features,) @ (in_features, out_features)
    #        -> (out_features,)
    #
    #
    #   3. Matrix @ Vector
    #        (batch, in_features) @ (in_features,)
    #        -> (batch,)
    #
    #
    #   4. Vector @ Vector
    #        (n,) @ (n,)
    #        -> scalar dot product
    #
    #
    # Important:
    # ----------
    # NumPy's np.matmul supports more advanced behavior, including batched
    # matrix multiplication with arrays of dimension 3 or higher. This small
    # autograd engine does NOT implement the backward pass for those cases.
    #
    # That is intentional. Supporting full NumPy matmul broadcasting would make
    # the backward pass much more complicated and would distract from the main
    # purpose of this project: understanding reverse-mode autodiff with
    # vectorized neural-network operations.
    #
    # If operands with ndim > 2 are passed here, the forward NumPy operation may
    # work, but the backward pass is not implemented, so we raise
    # NotImplementedError.
    #
    # For this project, the most important supported case is:
    #
    #        X @ W
    #
    # where:
    #
    #        X.shape == (batch_size, input_size)
    #        W.shape == (input_size, output_size)
    #
    # This is enough to build MLP layers efficiently.
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), "@")

        def _backward():
            # Common dense-layer case: (batch, in) @ (in, out).
            if self.data.ndim == 2 and other.data.ndim == 2:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            # Convenience cases for vectors. They are less important for MLPs,
            # but make the operator usable in small experiments.
            elif self.data.ndim == 1 and other.data.ndim == 2:
                self.grad += out.grad @ other.data.T
                other.grad += np.outer(self.data, out.grad)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                self.grad += np.outer(out.grad, other.data)
                other.grad += self.data.T @ out.grad
            elif self.data.ndim == 1 and other.data.ndim == 1:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            else:
                raise NotImplementedError(
                    "matmul backward currently supports only 1D/2D operands; "
                    "batched matmul with ndim > 2 is not implemented"
                )

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(self.data, 0), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def softmax_ce(self, targets):
        targets = np.asarray(targets, dtype=float)

        if self.data.ndim != 2:
            raise ValueError("softmax_ce expects logits with shape (batch, classes)")
        if targets.shape != self.data.shape:
            raise ValueError("targets must have the same shape as logits")

        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)

        n = targets.shape[0]
        loss = -(np.log(probs.clip(1e-7)) * targets).sum() / n
        out = Value(loss, (self,), "softmax_ce")

        def _backward():
            self.grad += ((probs - targets) / n) * out.grad

        out._backward = _backward
        return out, probs

    def exp(self):
        e = np.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad
            
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = 1
            for ax in axes:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def backward(self):
        """Backpropagate from this Value through the dynamically built graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=float)
        for v in reversed(topo):
            v._backward()

    def item(self):
        return self.data.item()

    def __float__(self):
        return float(self.data)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        if self.data.shape == ():
            data = self.data.item()
            grad = self.grad.item()
            return f"Value(data={data}, grad={grad})"
        return f"Value(shape={self.data.shape}, data={self.data}, grad={self.grad})"
