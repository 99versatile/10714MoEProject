from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=-1, keepdims=True)
        shifted = Z - max_Z
        exp_shifted = array_api.exp(shifted)
        sum_exp = array_api.sum(exp_shifted, axis=-1, keepdims=True)
        log_sum_exp = array_api.log(sum_exp) + max_Z
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        softmax_vals = exp(logsoftmax(lhs))
        
        sum_out_grad = summation(out_grad, axes=(-1,))
        
        sum_broadcasted = broadcast_to(reshape(sum_out_grad, (*sum_out_grad.shape, 1)), out_grad.shape)
        
        return add(out_grad, negate(multiply(softmax_vals, sum_broadcasted)))
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        # Normalize axes to be a tuple or None
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            M = Z.max(keepdims=True)  # shape (1,)
            M_shape = tuple([1] * len(Z.shape))
            M_broadcasted = M.compact().reshape(M_shape)
            shifted = Z - M_broadcasted
            exp_shifted = array_api.exp(shifted)
            sum_exp = array_api.sum(exp_shifted)  # shape (1,)
            result = array_api.log(sum_exp) + M
            return result
        else:
            # Normalize axes to positive integers
            if isinstance(self.axes, (tuple, list)):
                if len(self.axes) == 1:
                    axis = self.axes[0]
                    if axis < 0:
                        axis = axis + len(Z.shape)
                else:
                    # Multiple axes - need to normalize each
                    axis = tuple(ax if ax >= 0 else ax + len(Z.shape) for ax in self.axes)
            else:
                axis = self.axes
                if axis < 0:
                    axis = axis + len(Z.shape)
            # Reduce over specific axes
            M = Z.max(axis=axis, keepdims=True)
            M_broadcasted = M.broadcast_to(Z.shape)
            shifted = Z - M_broadcasted
            exp_shifted = array_api.exp(shifted)
            sum_exp = array_api.sum(exp_shifted, axis=axis, keepdims=False)
            # M has keepdims=True, need to reshape to match sum_exp
            M_squeezed = M.compact().reshape(sum_exp.shape)
            result = array_api.log(sum_exp) + M_squeezed
            return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)
        
        logsumexp_result = logsumexp(lhs, axes=axes)
        
        if axes is not None:
            shape = list(lhs.shape)
            for axis in axes:
                if axis < 0:
                    axis = axis + len(lhs.shape)
                shape[axis] = 1
            logsumexp_reshaped = reshape(logsumexp_result, tuple(shape))
            logsumexp_broadcasted = broadcast_to(logsumexp_reshaped, lhs.shape)
        else:
            logsumexp_broadcasted = broadcast_to(logsumexp_result, lhs.shape)
        
        unnormalized = exp(add(lhs, negate(logsumexp_broadcasted)))
        
        if axes is not None:
            sum_unnormalized = unnormalized
            for axis in sorted(axes, reverse=True):
                sum_unnormalized = summation(sum_unnormalized, axes=(axis,))
            
            shape_sum = list(lhs.shape)
            for axis in axes:
                if axis < 0:
                    axis = axis + len(lhs.shape)
                shape_sum[axis] = 1
            sum_reshaped = reshape(sum_unnormalized, tuple(shape_sum))
            sum_broadcasted = broadcast_to(sum_reshaped, lhs.shape)
            
            softmax_vals = divide(unnormalized, sum_broadcasted)
        else:
            total_sum = summation(unnormalized, axes=None)
            total_sum_broadcasted = broadcast_to(total_sum, lhs.shape)
            softmax_vals = divide(unnormalized, total_sum_broadcasted)
        
        if axes is not None:
            shape = list(lhs.shape)
            for axis in axes:
                if axis < 0:
                    axis = axis + len(lhs.shape)
                shape[axis] = 1
            out_grad_reshaped = reshape(out_grad, tuple(shape))
            out_grad_broadcasted = broadcast_to(out_grad_reshaped, lhs.shape)
            return multiply(out_grad_broadcasted, softmax_vals)
        else:
            return multiply(out_grad, softmax_vals)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)