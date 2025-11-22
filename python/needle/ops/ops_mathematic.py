"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad * rhs * power(lhs, rhs - 1), out_grad * log(lhs) * power(lhs, rhs))
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return (out_grad * self.scalar * lhs ** (self.scalar - 1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, out_grad * (-1) * lhs * rhs ** (-2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axis1, axis2 = a.ndim - 1, a.ndim - 2
        else:
            axis1, axis2 = self.axes
        axes = list(range(a.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return a.permute(tuple(axes))
        # return array_api.swapaxes(a, axis1, axis2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # The gradient of a transpose is a transpose with the same swap/permutation
        return (transpose(out_grad, axes=self.axes),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # If reshaping to the same shape, just compact and return
        if tuple(a.shape) == tuple(self.shape):
            return a.compact()
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return reshape(out_grad, lhs.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        lhs_shape = lhs.shape
        out_shape = out_grad.shape

        added_dim_num = len(out_shape) - len(lhs_shape)
        sum_axes = list(range(added_dim_num))

        for i, shape in enumerate(lhs_shape):
            if shape == 1 and out_shape[added_dim_num + i] != 1:
                sum_axes.append(added_dim_num + i)
        
        if len(sum_axes) != 0:
            o_grad = summation(out_grad, axes=tuple(sum_axes))
        else:
            o_grad = out_grad
            
        if o_grad.shape != lhs_shape:
            o_grad = reshape(o_grad, lhs_shape)

        return o_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.sum(a)
        else:
            # Normalize axes to positive indices
            if isinstance(self.axes, (tuple, list)):
                if len(self.axes) > 1:
                    # Multiple axes: normalize each and sum iteratively
                    normalized_axes = [ax if ax >= 0 else ax + len(a.shape) for ax in self.axes]
                    result = a
                    for axis in sorted(normalized_axes, reverse=True):
                        result = array_api.sum(result, axis=axis)
                    return result
                else:
                    # Single axis in tuple/list: extract and normalize
                    axis = self.axes[0]
                    if axis < 0:
                        axis = axis + len(a.shape)
                    return array_api.sum(a, axis=axis)
            else:
                # Single integer axis: normalize if negative
                axis = self.axes
                if axis < 0:
                    axis = axis + len(a.shape)
                return array_api.sum(a, axis=axis)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        
        if self.axes is None:
            new_shape = tuple(1 for _ in range(len(lhs.shape)))
        elif isinstance(self.axes, int): 
            new_shape = list(lhs.shape)
            axis = self.axes
            if self.axes < 0:
                axis = self.axes + len(lhs.shape)
            new_shape[axis] = 1
        else:
            new_shape = list(lhs.shape)
            for axis in self.axes:
                if axis < 0:
                    axis = axis + len(lhs.shape)
                new_shape[axis] = 1
            new_shape = tuple(new_shape)
        
        reshaped_grad = reshape(out_grad, new_shape)
        
        return broadcast_to(reshaped_grad, lhs.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        grad_lhs = matmul(out_grad, transpose(rhs, axes=(-2, -1)))
        grad_rhs = matmul(transpose(lhs, axes=(-2, -1)), out_grad)
        
        if grad_lhs.shape == lhs.shape and grad_rhs.shape == rhs.shape:
            return (grad_lhs, grad_rhs)
        else:
            sum_axis_lhs = tuple(range(len(grad_lhs.shape) - len(lhs.shape)))
            sum_axis_rhs = tuple(range(len(grad_rhs.shape) - len(rhs.shape)))
            grad_lhs = summation(grad_lhs, axes=sum_axis_lhs)
            grad_rhs = summation(grad_rhs, axes=sum_axis_rhs)

            grad_lhs = reshape(grad_lhs, lhs.shape)
            grad_rhs = reshape(grad_rhs, rhs.shape)

            return (grad_lhs, grad_rhs)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return divide(out_grad, lhs)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return multiply(out_grad, exp(lhs))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        lhs_data = lhs.realize_cached_data()
        return out_grad * (lhs_data > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class SiLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.sigmoid_a = 1 / (1 + array_api.exp(-a))
        return a * self.sigmoid_a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return out_grad * (self.sigmoid_a + lhs * (1 - self.sigmoid_a))
        ### END YOUR SOLUTION


def silu(a):
    return SiLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (array_api.exp(a) - array_api.exp(-a)) /  (array_api.exp(a) + array_api.exp(-a))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return out_grad - out_grad * (node ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        base_shape = args[0].shape
        n_arrays = len(args)
        
        new_shape = list(base_shape)
        new_shape.insert(self.axis, n_arrays)
        new_shape = tuple(new_shape)
        
        out = array_api.empty(new_shape, device=args[0].device)
        
        for i, arr in enumerate(args):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            out[tuple(slices)] = arr
        
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n_splits = A.shape[self.axis]
        
        result = []
        for i in range(n_splits):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            sliced = A[tuple(slices)]
            new_shape = list(sliced.shape)
            new_shape.pop(self.axis)
            result.append(sliced.compact().reshape(tuple(new_shape)))
        
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        """
        note that these should be extremely short.
        """
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        
        for ax in self.axes:
            if ax < len(new_shape):
                new_shape[ax] = new_shape[ax] * (self.dilation + 1)
        
        dilated = array_api.full(new_shape, 0.0, device=a.device)
        
        slices = [slice(None)] * len(new_shape)
        for ax in self.axes:
            if ax < len(new_shape):
                slices[ax] = slice(0, new_shape[ax], self.dilation + 1)
        
        dilated[tuple(slices)] = a
        
        return dilated
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        old_shape = a.shape
        new_shape = list(a.shape)

        for ax in self.axes:
            if ax < len(new_shape):
                new_shape[ax] = new_shape[ax] // (self.dilation + 1)

        slices = [slice(None)] * len(old_shape)
        for ax in self.axes:
            if ax < len(old_shape):
                slices[ax] = slice(0, old_shape[ax], self.dilation + 1)

        undilated = a[tuple(slices)]

        return undilated
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        """
        2D convolution
        accept tensors in the NHWC format
        and weights in the format of 
        (kernel_size, kernel_size, input_channels, output_channels)
        """
        A_zero_padded = A.pad( ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)) )
        
        N, H, W, C_in = A_zero_padded.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A_zero_padded.strides
        
        # Calculate output dimensions with stride
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        # im2col using as_strided with stride
        inner_dim = K * K * C_in
        im2col = A_zero_padded.as_strided(
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        ).compact().reshape((N * H_out * W_out, inner_dim))
        
        # Matrix multiplication
        out = im2col @ B.compact().reshape((K * K * C_in, C_out))
        
        # Reshape to output
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out_grad is (N, H_out, W_out, C_out)
        A, B = node.inputs  # A: input (N,H,W,C_in), B: weight (K,K,C_in,C_out)
        K = B.shape[0]
        
        # ===== Gradient w.r.t. input A =====
        # X.grad ≈ conv(out_grad, W) with appropriate modifications
        B_flipped = flip(B, (0, 1))  
        B_flipped = transpose(B_flipped, (2, 3))
        
        if self.stride > 1:
            out_grad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        else:
            out_grad_dilated = out_grad
        
        padding_A = K - 1 - self.padding
        
        grad_A = conv(out_grad_dilated, B_flipped, stride=1, padding=padding_A)
        
        # ===== Gradient w.r.t. weight B =====
        # Compute dL/dW using im2col approach
        # grad_W[k1,k2,c_in,c_out] = sum_{n,h_out,w_out} grad_Y[n,h_out,w_out,c_out] * X[n, h_out*s+k1, w_out*s+k2, c_in]
        
        # 1. Pad input A
        A_padded = A.realize_cached_data().pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )
        
        # 2. Dilate out_grad if stride > 1
        if self.stride > 1:
            out_grad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        else:
            out_grad_dilated = out_grad
        out_grad_data = out_grad_dilated.realize_cached_data()
        
        # 3. Use im2col to extract patches from A
        N, H_pad, W_pad, C_in = A_padded.shape
        N_g, H_out_d, W_out_d, C_out = out_grad_data.shape
        Ns, Hs, Ws, Cs = A_padded.strides
        
        # Extract K x K patches
        inner_dim = K * K * C_in
        im2col = A_padded.as_strided(
            shape=(N, H_out_d, W_out_d, K, K, C_in),
            strides=(Ns, Hs, Ws, Hs, Ws, Cs)
        ).compact().reshape((N * H_out_d * W_out_d, inner_dim))
        
        # 4. Reshape out_grad for matrix multiplication
        out_grad_reshaped = out_grad_data.compact().reshape((N * H_out_d * W_out_d, C_out))
        
        # 5. Compute grad_B via matrix multiplication
        # im2col^T @ out_grad = (K*K*C_in, N*H*W) @ (N*H*W, C_out) = (K*K*C_in, C_out)
        # Use NDArray transpose
        im2col_T = im2col.permute((1, 0))  # Transpose to (inner_dim, N*H*W)
        grad_B_flat = im2col_T @ out_grad_reshaped  # (K*K*C_in, C_out)
        
        # 6. Reshape to (K, K, C_in, C_out) and wrap in Tensor
        from needle import Tensor
        grad_B = Tensor.make_const(grad_B_flat.reshape((K, K, C_in, C_out)))
        
        return grad_A, grad_B
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

class Topk(TensorOp):
    def __init__(self, k: int, axis: int = -1):
        self.k = k
        self.axis = axis

    def compute(self, a):
        """
        Compute the topk values and indices of a tensor.
        Returns: tuple of (values, indices)
        values: top k values along axis
        indices: indices of top k values along axis
        """
        k = self.k
        axis = self.axis
        
        # Handle negative axis
        if axis < 0:
            axis = len(a.shape) + axis

        # Use argpartition to get k largest indices efficiently (not sorted)
        # indices = a.argpartition(-k, axis=axis)
        indices = array_api.argpartition(a, -k, axis=axis)

        # Get only the last k indices (=k largest indices)
        indices = array_api.take(indices, array_api.arange(-k, 0), axis=axis)

        # Sort these k indices by their values
        # Gather values at these indices first
        values = array_api.take_along_axis(a, indices, axis=axis)

        # Get sorting permutation within these k elements -> efficient since we're sorting only k elements
        sorted_positions = array_api.argsort(-values, axis=axis)  # negative for descending order
        
        # Apply sorting to both values and indices
        sorted_values = array_api.take_along_axis(values, sorted_positions, axis=axis)
        sorted_indices = array_api.take_along_axis(indices, sorted_positions, axis=axis)

        return sorted_values, sorted_indices

    def gradient(self, out_grad, node):
        """
        Gradient: scatter out_grad back to positions indicated by indices.
        Only top-k positions receive gradients, rest are zero.
        
        out_grad: gradient w.r.t output values
        lhs: original input tensor
        node.realize_cached_data(): (values, indices) tuple from forward pass
        """
        # Get the original input tensor
        lhs, = node.inputs
        # Get the topk indices from forward pass
        _, indices = node.realize_cached_data()

        # Create zero gradient of input shape
        input_shape = lhs.shape
        grad_input = array_api.zeros(input_shape, device=node.device)

        # Scatter out_grad to positions indicated by indices
        grad_input = array_api.scatter_along_axis(
            grad_input, 
            indices, 
            out_grad, 
            axis=self.axis
        )
        
        return grad_input

def topk(a, k, axis=-1):
    """
    Compute the topk values and indices of a tensor.    
    
    Args:
        a: Input tensor
        k: Number of top elements
        axis: Axis along which to compute topk
        
    Returns:
        Tuple of (values, indices) tensors
    """
    return Topk(k, axis)(a)