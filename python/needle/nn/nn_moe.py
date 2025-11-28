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
    SiLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential,
    SoftmaxLoss,
)
from .nn_transformer import AttentionLayer
from needle.backend_selection import array_api
from typing import Any


class TopKRouter(Module):
    def __init__(self, num_experts: int, d_model: int, topk: int, device: Any | None = None, dtype: str = "float32"):
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


class SwiGLU(Module):
    def __init__(self, in_features: int, hidden_size: int, device: Any | None = None, dtype: str = "float32"):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.linear1 = Linear(in_features, hidden_size, device=device, dtype=dtype)
        self.silu = SiLU()
        self.linear2 = Linear(in_features, hidden_size, device=device, dtype=dtype)
        self.linear3 = Linear(hidden_size, in_features, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear3(self.linear1(x) * self.silu(self.linear2(x)))

class TopKMoE(Module):
    def __init__(self, num_experts: int, d_model: int, topk: int, hidden_size: int, device: Any | None = None, dtype: str = "float32"):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.topk = topk
        self.router = TopKRouter(num_experts, d_model, topk, device=device, dtype=dtype)

        self.experts = [SwiGLU(d_model, hidden_size, device=device, dtype=dtype)
                        for _ in range(num_experts)]

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
        x: [batch, seq, d_model]
        Returns:
        output: [batch, seq, d_model]
        """
        # Routing Phase
        router_probs, topk_indices, logits = self.router(x)
        # router_probs: [b, s, num_experts]
        # topk_indices: [b, s, topk]

        b, s, _ = topk_indices.shape

        # Dispatch Phase (token -> expert)
        # For each expert gather tokens
        expert_inputs = [[] for _ in range(self.num_experts)]
        positions = [[] for _ in range(self.num_experts)]  # for scatter restore

        for batch in range(b):
            for seq in range(s):
                for k in range(self.topk):
                    e = topk_indices[batch, seq, k].numpy().item()
                    expert_inputs[e].append(x[batch, seq])
                    positions[e].append((batch, seq, k))

        # Convert lists to tensors
        expert_inputs = [
            Tensor.stack(tokens) if tokens else None
            for tokens in expert_inputs
        ]
        
        # Forward Phase (expert -> expert outputs)
        expert_outputs = []
        for e in range(self.num_experts):
            if expert_inputs[e] is None:
                # Skip non-routed experts
                expert_outputs.append(None)
            else:
                # Forward pass through routed experts
                expert_outputs.append(self.experts[e](expert_inputs[e]))

        # Combine Phase (expert outputs -> output)
        output = init.zeros_like(x)

        for e in range(self.num_experts):
            if expert_outputs[e] is None:
                continue
            out_e = expert_outputs[e]
            pos_e = positions[e]
            for i, (batch, seq, k) in enumerate(pos_e):
                prob = router_probs[batch, seq, topk_indices[batch,seq,k]]
                output[batch, seq] += prob * out_e[i]

        return output


class TopKMoETransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        num_experts: int,
        topk: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.q_features = q_features
        self.num_head = num_head
        self.dim_head = dim_head
        self.hidden_size = hidden_size
        self.attn = AttentionLayer(q_features, num_head, dim_head, device=device, dtype=dtype, dropout=dropout, causal=causal)
        self.dropout1 = Dropout(dropout)
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.topk_moe = TopKMoE(num_experts, q_features, topk, hidden_size, device=device, dtype=dtype)
        self.dropout2 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        a = self.attn(x)
        a = self.dropout1(a)
        x = x + a

        # FFN block with Pre-LN
        # Reshape for LayerNorm1d: (B, T, D) -> (B*T, D)
        f = x.reshape((batch_size * seq_len, x_dim))
        f = self.norm(f)
        # Keep f in 2D for TopKMoE
        f = self.topk_moe(f)
        f = self.dropout2(f)
        # Reshape back to 3D before residual connection
        f = f.reshape((batch_size, seq_len, x_dim))
        y = x + f
        ### END YOUR SOLUTION

        return y


class TopKMoETransformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        num_experts: int,
        topk: int,
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.layers = Sequential(*[TopKMoETransformerLayer(embedding_size, num_head, dim_head, hidden_size, num_experts, topk, dropout=dropout, causal=causal, device=device, dtype=dtype) for _ in range(num_layers)])
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, _ = x.shape
        
        x = self.layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)