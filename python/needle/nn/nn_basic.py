"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            return out + ops.broadcast_to(self.bias, out.shape)
        else:
            return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, [X.shape[0], -1])
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.silu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            result = module(x)

            if isinstance(result, tuple):
                x = result[0]
            else:
                x = result
        
        if isinstance(result, tuple):
            return result
        else:
            return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        logsumexp_vals = ops.logsumexp(logits, axes=(-1,))
        one_hot_y = init.one_hot(logits.shape[-1], y, device=logits.device, dtype=logits.dtype)
        
        correct_logits = ops.summation(logits * one_hot_y, axes=(-1,))
        sample_losses = logsumexp_vals - correct_logits
        return ops.summation(sample_losses, axes=None) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # During training: use batch statistics and update running statistics
            batch_mean = ops.summation(x, axes=(0,)) / x.shape[0]
            mean_broadcast = ops.broadcast_to(ops.reshape(batch_mean, (1, x.shape[1])), x.shape)
            x_centered = ops.add(x, ops.negate(mean_broadcast))
            batch_var = ops.summation(ops.power_scalar(x_centered, 2), axes=(0,)) / x.shape[0]
            
            # Update running statistics (detached from computation graph)  
            self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Use batch statistics for normalization
            var_broadcast = ops.broadcast_to(ops.reshape(batch_var, (1, x.shape[1])), x.shape)
        else:
            # During inference: use running statistics
            mean_broadcast = ops.broadcast_to(ops.reshape(self.running_mean, (1, x.shape[1])), x.shape)
            x_centered = ops.add(x, ops.negate(mean_broadcast))
            var_broadcast = ops.broadcast_to(ops.reshape(self.running_var, (1, x.shape[1])), x.shape)
        
        var_eps = ops.add_scalar(var_broadcast, self.eps)
        std = ops.power_scalar(var_eps, 0.5)

        # apply: (x - mean) / sqrt(variance + eps)
        normalized = ops.divide(x_centered, std)

        # Reshape weight and bias from (num_features,) to (1, num_features) before broadcasting
        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, (1, x.shape[1])), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, (1, x.shape[1])), x.shape)
        y = ops.add(ops.multiply(weight_broadcast, normalized), bias_broadcast)
        
        return y
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Works for (B, D), (B, T, D), etc. Normalize over last dim.
        dim = self.dim
        assert x.shape[-1] == dim, "LayerNorm1d: last dim must equal self.dim"

        # mean over last dimension
        mean = ops.summation(x, axes=(-1,)) / dim          # shape: x.shape[:-1]
        mean = ops.reshape(mean, x.shape[:-1] + (1,))      # (..., 1)
        mean = ops.broadcast_to(mean, x.shape)             # (..., D)

        # variance over last dimension
        x_centered = ops.add(x, ops.negate(mean))
        x_squared = ops.power_scalar(x_centered, 2)
        var = ops.summation(x_squared, axes=(-1,)) / dim   # shape: x.shape[:-1]
        var = ops.reshape(var, x.shape[:-1] + (1,))
        var = ops.broadcast_to(var, x.shape)

        var_eps = ops.add_scalar(var, self.eps)
        std = ops.power_scalar(var_eps, 0.5)
        normalized = ops.divide(x_centered, std)

        # scale + shift: reshape params to (1,...,1, dim)
        param_shape = (1,) * (len(x.shape) - 1) + (dim,)
        weight = ops.reshape(self.weight, param_shape)
        bias = ops.reshape(self.bias, param_shape)

        y = ops.add(ops.multiply(weight, normalized), bias)
        return y
        ### END YOUR SOLUTION



class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            keep_mask = init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype)
            scale_factor = 1.0 / (1 - self.p)
            return ops.mul_scalar(ops.multiply(keep_mask, x), scale_factor)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(self.fn(x), x)
        ### END YOUR SOLUTION
