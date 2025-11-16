import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
# np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype
        conv1 = nn.Conv(3, 16, kernel_size=7, stride=4, device=device, dtype=dtype)
        bn1 = nn.BatchNorm2d(16, device=device, dtype=dtype)
        relu1 = nn.ReLU()
        
        conv2 = nn.Conv(16, 32, kernel_size=3, stride=2, device=device, dtype=dtype)
        bn2 = nn.BatchNorm2d(32, device=device, dtype=dtype)
        relu2 = nn.ReLU()

        conv3 = nn.Conv(32, 32, kernel_size=3, stride=1, device=device, dtype=dtype)
        bn3 = nn.BatchNorm2d(32, device=device, dtype=dtype)
        relu3 = nn.ReLU()

        conv4 = nn.Conv(32, 32, kernel_size=3, stride=1, device=device, dtype=dtype)
        bn4 = nn.BatchNorm2d(32, device=device, dtype=dtype)
        relu4 = nn.ReLU()

        residual1 = nn.Residual(nn.Sequential(conv3, bn3, relu3, conv4, bn4, relu4))
        
        conv5 = nn.Conv(32, 64, kernel_size=3, stride=2, device=device, dtype=dtype)
        bn5 = nn.BatchNorm2d(64, device=device, dtype=dtype)
        relu5 = nn.ReLU()

        conv6 = nn.Conv(64, 128, kernel_size=3, stride=2, device=device, dtype=dtype)
        bn6 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        relu6 = nn.ReLU()

        conv7 = nn.Conv(128, 128, kernel_size=3, stride=1, device=device, dtype=dtype)
        bn7 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        relu7 = nn.ReLU()

        conv8 = nn.Conv(128, 128, kernel_size=3, stride=1, device=device, dtype=dtype)
        bn8 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        relu8 = nn.ReLU()

        residual2 = nn.Residual(nn.Sequential(conv7, bn7, relu7, conv8, bn8, relu8))
        
        flatten = nn.Flatten()

        linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        relu9 = nn.ReLU()

        linear2 = nn.Linear(128, 10, device=device, dtype=dtype)

        self.net = nn.Sequential(conv1, bn1, relu1, conv2, bn2, relu2, residual1, conv5, bn5, relu5, conv6, bn6, relu6, residual2, flatten, linear1, relu9, linear2)
        ### END YOUR SOLUTION ###

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.net(x)
        ### END YOUR SOLUTION ###


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise ValueError(f"Invalid sequence model: {seq_model}")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x)
        x, h = self.seq_model(x, h)
        seq_len, bs, hidden_size = x.shape
        x = x.reshape((seq_len * bs, hidden_size))
        x = self.linear(x)
        return x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
