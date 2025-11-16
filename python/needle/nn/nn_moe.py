from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)
from needle.backend_selection import array_api

class MoERouter(Module):
    def __init__(self, num_experts, d_model, topk, device=None, dtype="float32"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.topk = topk
        self.device = device
        self.dtype = dtype
        self.gate = Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)

    def softmax(self, logit):
        """
        The softmax function for the MoE router.
        Subtract the max value from the logit to avoid overflow.
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom    
    
    def forward(self, x):
        """
        The forward pass for the MoE router.
        """
        ### BEGIN YOUR SOLUTION
        # Compute the logits for the experts
        logits = self.gate(x)
        
        # Compute the topk values and indices
        topk_logits, topk_indices = ops.topk(logits, self.topk)

        # Compute sparse softmax
        zeros = Tensor(array_api.full_like(logits, float('-inf')))
        sparse_logits = zeros.scatter(-1, topk_indices, topk_logits)
        router_probs = self.softmax(sparse_logits)
        
        return router_probs, topk_indices, logits