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
from needle.backend_selection import array_api
from typing import Any, Optional
from .nn_transformer import Transformer
from .nn_moe import TopKMoETransformer
from .nn_basic import Residual


class Perplexity(Module):
    def __init__(self, loss_fn: SoftmaxLoss):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        # Cross-entropy (mean over batch)
        loss = self.loss_fn(logits, y)

        # Perplexity = exp(cross entropy)
        return ops.exp(loss)


class TokenEmbeddings(Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        max_position_embeddings: int,
        padding_idx: int | None = None,
        word_embed_proj_dim: int | None = None,
        learnable: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ):
        """
        GPT-2 Learnable Token and Position Embeddings.
        If max_position_embeddings <= 0, there's no position embeddings
        We embed to word_embe_proj_dim dimension then project up to embed_dim
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx, device=device, dtype=dtype
            )
            self.project_in = None
        else:
            self.word_embeddings = Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
                device=device,
                dtype=dtype
            )
            self.project_in = Linear(
                word_embed_proj_dim, embed_dim, bias=False, device=device, dtype=dtype
            )
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = Embedding(
                max_position_embeddings, embed_dim, device=device, dtype=dtype
            )

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen) or None
        Returns: (batch, seqlen, embed_dim)
        """
        batch_size, seqlen = input_ids.shape
        
        # Embedding expects (seq_len, bs), so transpose input_ids
        input_ids_transposed = ops.transpose(input_ids, axes=(1, 0))  # (seqlen, batch_size)
        embeddings = self.word_embeddings(input_ids_transposed)  # (seqlen, batch_size, embed_dim)
        
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        
        if self.max_position_embeddings > 0:
            if position_ids is None:
                # Create position ids: (seqlen, batch_size) with broadcasted values
                positions = np.arange(seqlen, dtype=np.int32).reshape((seqlen, 1))
                positions_tensor = Tensor(positions, device=input_ids.device, dtype=input_ids.dtype)
                position_ids_transposed = ops.broadcast_to(positions_tensor, (seqlen, batch_size))
            else:
                # Transpose position_ids from (batch, seqlen) to (seqlen, batch_size)
                position_ids_transposed = ops.transpose(position_ids, axes=(1, 0))
            
            position_embeddings = self.position_embeddings(position_ids_transposed)  # (seqlen, batch_size, embed_dim)
            embeddings = embeddings + position_embeddings
        
        # Transpose back to (batch_size, seqlen, embed_dim) 
        embeddings = ops.transpose(embeddings, axes=(1, 0, 2))
        return embeddings


class LMBackbone(Module):
    def __init__(
        self, 
        embedding_size: int, 
        vocab_size: int, 
        max_position_embeddings: int, 
        learnable_word_embeddings: bool,
        n_layers: int,
        block_type: str,
        hidden_size: int,
        num_head: int,
        dim_head: int,
        dropout: float,
        causal: bool,
        batch_first: bool,
        sequence_len: int,
        resid_dropout: float,
        layer_norm_epsilon: float,
        num_experts: Optional[int] = None,
        topk: Optional[int] = None,
        device: Any | None = None,
        dtype: str = "float32"
    ):
        super().__init__()
        
        # Token embeddings
        self.embeddings = TokenEmbeddings(
            embed_dim=embedding_size, 
            vocab_size=vocab_size, 
            max_position_embeddings=max_position_embeddings,
            learnable_word_embeddings=learnable_word_embeddings,
            device=device,
            dtype=dtype
        )
        
        # Create transformer layers with hybrid support
        self.modules_list = []
        for i in range(n_layers):
            # Use the specified block type for homogeneous models (backward compatibility)
            if block_type == 'Transformer':
                layer = Transformer(
                    embedding_size=embedding_size, 
                    hidden_size=hidden_size, 
                    num_layers=n_layers,
                    num_head=num_head,
                    dim_head=dim_head,
                    dropout=dropout,
                    causal=causal,
                    batch_first=batch_first,
                    sequence_len=sequence_len, 
                    device=device,
                    dtype=dtype,
                )
            elif block_type == 'TopkMoETransformer':
                layer = TopKMoETransformer(
                    embedding_size=embedding_size, 
                    hidden_size=hidden_size, 
                    num_layers=n_layers,
                    num_experts=num_experts, 
                    topk=topk,
                    vocab_size=vocab_size, 
                    max_position_embeddings=max_position_embeddings, 
                    learnable_word_embeddings=learnable_word_embeddings,
                    num_head=num_head,
                    dim_head=dim_head,
                    dropout=dropout,
                    causal=causal,
                    batch_first=batch_first,
                    sequence_len=sequence_len,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            
            self.modules_list.append(layer)
        
        self.layers = Sequential(*self.modules_list)
        # Final normalization and dropout
        self.drop_f = Dropout(resid_dropout)
        self.ln_f = LayerNorm1d(embedding_size, eps=layer_norm_epsilon, device=device, dtype=dtype)
        
        # # Initialize weights  -> Also done in LanguageModel, remove to avoid double initialization
        # self.apply(partial(_init_weights, n_layers=config.n_layers, block_type=config.block_type))

    def forward(self, input_ids, position_ids=None):
        batch_size_input, seq_len_input = input_ids.shape
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
        )
        batch_size_hidden, seq_len_hidden, embedding_size_hidden = hidden_states.shape
        
        assert batch_size_hidden == batch_size_input and seq_len_hidden == seq_len_input, "Batch size and sequence length must match between input and hidden states"
        
        for layer in self.layers:
            # Transformer and TopKMoETransformer return (x, h) tuple
            hidden_states, _ = layer(hidden_states)
            assert hidden_states.shape == (batch_size_hidden, seq_len_hidden, embedding_size_hidden), "Hidden states shape must match between layers"
        # Final LayerNorm computed in float32 for stability, cast back to original dtype
        hidden_states = self.ln_f(self.drop_f(hidden_states).to("float32")).to(hidden_states.dtype)
        return hidden_states


class LanguageModel(Module):
    """
    Main Language Model class that supports arbitrary mixer configurations
    """
    def __init__(
        self, 
        embedding_size: int, 
        vocab_size: int, 
        max_position_embeddings: int, 
        learnable_word_embeddings: bool, 
        n_layers: int, 
        block_type: str, 
        hidden_size: int, 
        num_head: int, 
        dim_head: int, 
        dropout: float, 
        causal: bool, 
        batch_first: bool, 
        sequence_len: int, 
        resid_dropout: float, 
        layer_norm_epsilon: float, 
        pad_vocab_size_multiple: int,
        label_smoothing: float = 0.0,
        tie_word_embeddings: bool = True,
        device: Any | None = None, 
        dtype: str = "float32",
        num_experts: Optional[int] = None, 
        topk: Optional[int] = None, 
    ):
        
        super().__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        
        # Ensure vocab size is properly padded
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        # Create backbone and language modeling head
        self.backbone = LMBackbone(
            embedding_size=embedding_size, 
            vocab_size=vocab_size, 
            max_position_embeddings=max_position_embeddings, 
            learnable_word_embeddings=learnable_word_embeddings, 
            n_layers=n_layers, block_type=block_type, 
            hidden_size=hidden_size, 
            num_head=num_head, 
            dim_head=dim_head, 
            dropout=dropout, 
            causal=causal, 
            batch_first=batch_first, 
            sequence_len=sequence_len, 
            resid_dropout=resid_dropout, 
            layer_norm_epsilon=layer_norm_epsilon, 
            num_experts=num_experts, 
            topk=topk,
            device=device,
            dtype=dtype,
        )
        self.label_smoothing = label_smoothing
        self.lm_head = Linear(embedding_size, vocab_size, bias=False, device=device, dtype=dtype)

        # Optionally tie weights between input embeddings and output head BEFORE initialization
        # This ensures the tied weights get properly initialized together
        if tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
            self, 
            input_ids, 
            labels=None, 
            position_ids=None, 
            return_logits: bool = True,
            return_hidden_states: bool = False,
        ):
        """
        Forward pass with optional loss computation
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            position_ids: Position ids [batch_size, seq_len] 
            
        Returns:
            dict containing logits and optionally loss
        """
        
        batch_size, seq_len = input_ids.shape

        hidden_states = self.backbone(input_ids, position_ids=position_ids)

        assert hidden_states.shape == (batch_size, seq_len, self.embedding_size), "Hidden states shape must match (batch_size, seq_len, embedding_size)"

        logits = self.lm_head(hidden_states)

        assert logits.shape == (batch_size, seq_len, self.vocab_size), "Logits shape must match (batch_size, seq_len, vocab_size)"
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            smoothing = self.label_smoothing
            loss_fct = SoftmaxLoss(label_smoothing=smoothing)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Compute CE loss in float32 for numerical stability under mixed precision
            loss = loss_fct(shift_logits.float(), shift_labels)
            # Save memory by not returning logits unless explicitly requested
            if not return_logits:
                logits = None
        else:
            # If no labels and logits are not requested, free logits to save memory
            if not return_logits:
                logits = None
        
        result = {
            'loss': loss
        }
        if return_logits:
            result['logits'] = logits
        if return_hidden_states:
            result['hidden_states'] = hidden_states
        return result
    