"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Current implementation (line 17)
        return ops.power_scalar(ops.add_scalar(ops.exp(-x), 1), -1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        self.W_ih = Parameter(init.rand(input_size, hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
            self.bias_hh = Parameter(init.rand(hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
        if self.nonlinearity == 'tanh':
            activation = ops.tanh
        elif self.nonlinearity == 'relu':
            activation = ops.relu
        else:
            raise ValueError(f"Unidentified nonlinearity: {self.nonlinearity}")
        
        result = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)

        if self.bias_ih is not None:
            result = result + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(result.shape)
        if self.bias_hh is not None:
            result = result + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(result.shape)

        return activation(result)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.rnn_cells = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.rnn_cells.append(RNNCell(input_dim, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Split h0 once and reshape to remove extra dimension
        h_splits = ops.split(h0, axis=0)
        h_list = [ops.reshape(h_splits[i], (bs, self.hidden_size)) for i in range(self.num_layers)]
        
        # Split X once before the loop for efficiency
        X_splits = ops.split(X, axis=0)
    
        outputs = []
        
        # Iterate through sequence
        for t in range(seq_len):
            # Get the t-th timestep and reshape to remove extra dimension
            x_t = ops.reshape(X_splits[t], (bs, self.input_size))
            
            # Process through each layer
            for layer_idx in range(self.num_layers):
                h_list[layer_idx] = self.rnn_cells[layer_idx](x_t, h_list[layer_idx])
                x_t = h_list[layer_idx]  # Output of this layer becomes input to next
            
            outputs.append(x_t)  # Collect output from last layer
        
        # Stack outputs: (seq_len, bs, hidden_size)
        output = ops.stack(outputs, axis=0)
        
        # Stack hidden states: (num_layers, bs, hidden_size)
        h_n = ops.stack(h_list, axis=0)
        
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.sigmoid = Sigmoid()
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
            self.bias_hh = Parameter(init.rand(4*hidden_size, device=device, dtype=dtype, low=-1/((hidden_size)**0.5), high=1/((hidden_size)**0.5)))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        
        bs = X.shape[0]

        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        gates = ops.matmul(X, self.W_ih) + ops.matmul(h0, self.W_hh)
        
        if self.bias:
            gates = gates + self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size))
            gates = gates + self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size))

        gates_reshaped = ops.reshape(gates, (bs, 4, self.hidden_size))

        i, f, g, o = ops.split(gates_reshaped, axis=1)

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = ops.tanh(g)
        o = self.sigmoid(o)

        ct = f * c0 + i * g
        ht = o * ops.tanh(ct)

        return ht, ct
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.lstm_cells = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_cells.append(LSTMCell(input_dim, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        h_splits = ops.split(h0, axis=0)
        c_splits = ops.split(c0, axis=0)
        h_list = [ops.reshape(h_splits[i], (bs, self.hidden_size)) for i in range(self.num_layers)]
        c_list = [ops.reshape(c_splits[i], (bs, self.hidden_size)) for i in range(self.num_layers)]

        X_splits = ops.split(X, axis=0)

        outputs = []

        for t in range(seq_len):
            x_t = ops.reshape(X_splits[t], (bs, self.input_size))
            for layer_idx in range(self.num_layers):
                h_list[layer_idx], c_list[layer_idx] = self.lstm_cells[layer_idx](x_t, (h_list[layer_idx], c_list[layer_idx]))
                x_t = h_list[layer_idx]
            outputs.append(x_t)

        output = ops.stack(outputs, axis=0)

        h_n = ops.stack(h_list, axis=0)
        c_n = ops.stack(c_list, axis=0)

        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype, mean=0, std=1))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        one_hot_2d = one_hot.reshape((seq_len * bs, self.num_embeddings))
        result = ops.matmul(one_hot_2d, self.weight)  # (seq_len*bs, embedding_dim)
        return result.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION