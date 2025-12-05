from ..autograd import Op, Tensor, TensorTuple, Value, TensorOp, TensorTupleOp
import needle.init as init
from ..backend_selection import array_api

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class Topk(TensorTupleOp):
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