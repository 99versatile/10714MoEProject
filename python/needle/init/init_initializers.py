import math
from .init_basic import *
from typing import Any, Optional


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    uniform = ndl.init.rand(fan_in, fan_out, low=-a, high=a, **kwargs)

    return uniform
    # output: fan_in x fan_out 2D tensor
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    normal = ndl.init.randn(fan_in, fan_out, mean=0, std=std, **kwargs)

    return normal
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: Optional[int], fan_out: Optional[int], nonlinearity: str = "relu", shape: Optional[tuple] = None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    """
    In particular, it should support a new shape argument which is then passed to, 
    e.g., the underlying rand function. Specifically, if the argument shape is not None, 
    then ignore fan_in and fan_out, and use the value of shape for initializations instead.
    """
    if shape is not None:
        gain = math.sqrt(2)
        bound = gain * math.sqrt(3 / fan_in)
        uniform = ndl.init.rand(*shape, low=-bound, high=bound, **kwargs)
        return uniform
    else:
        gain = math.sqrt(2)
        bound = gain * math.sqrt(3 / fan_in)
        uniform = ndl.init.rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
        return uniform
    ### END YOUR SOLUTION

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    normal = ndl.init.randn(fan_in, fan_out, mean=0, std=std, **kwargs)

    return normal
    ### END YOUR SOLUTION