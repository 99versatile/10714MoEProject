"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_img = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        img_data = f.read()
        images = np.frombuffer(img_data, dtype=np.uint8).reshape(num_img, num_rows * num_cols).astype(np.float32)
        images = images / 255.0

    with gzip.open(label_filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_img = struct.unpack('>I', f.read(4))[0]

        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)

    return images, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))  # shape: (batch_size,)
    true_class_logits = ndl.summation(Z * y_one_hot, axes=(1,))  # shape: (batch_size,)
    losses = log_sum_exp - true_class_logits  # shape: (batch_size,)
    avg_softmax_loss = ndl.summation(losses) / Z.shape[0]
    
    return avg_softmax_loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    for i in range(0, X.shape[0], batch): # W1 shape (input_dim, hidden_dim), W2 shape (hidden_dim, num_classes)
        minibatch_X = ndl.Tensor(X[i:i+batch])  # shape (batch, input_dim)
        minibatch_y = y[i:i+batch]  # shape (batch,)
        
        one_hot_y = np.zeros((batch, W2.shape[1]))  # shape (batch, num_classes)
        one_hot_y[np.arange(batch), minibatch_y] = 1
        one_hot_y = ndl.Tensor(one_hot_y)

        minibatch_Z1 = ndl.relu(ndl.matmul(minibatch_X, W1)) # shape (batch, hidden_dim)
        minibatch_Z2 = ndl.matmul(minibatch_Z1, W2) # shape (batch, num_classes)

        exp_Z2 = ndl.exp(minibatch_Z2)  # Use pre-computed Z2
        sum_exp_Z2 = ndl.summation(exp_Z2, axes=(1,))  # shape (batch,)
        sum_exp_Z2_reshaped = ndl.reshape(sum_exp_Z2, (sum_exp_Z2.shape[0], 1))  # shape (batch, 1)
        softmax = ndl.divide(exp_Z2, sum_exp_Z2_reshaped)  
        minibatch_G2 = ndl.add(softmax, ndl.negate(one_hot_y))  # softmax - one_hot_y
        relu_mask = ndl.Tensor((minibatch_Z1.numpy() > 0).astype(np.float32))
        minibatch_G1 = ndl.multiply(relu_mask, ndl.matmul(minibatch_G2, ndl.transpose(W2))) # shape (batch, hidden_dim)

        gradient_W1 = ndl.divide_scalar(ndl.matmul(ndl.transpose(minibatch_X), minibatch_G1), batch)
        gradient_W2 = ndl.divide_scalar(ndl.matmul(ndl.transpose(minibatch_Z1), minibatch_G2), batch)     

        W1 -= ndl.mul_scalar(gradient_W1, lr)
        W2 -= ndl.mul_scalar(gradient_W2, lr)
    
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    correct = 0
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        X, y = batch
        X, y = ndl.Tensor(X, device=model.device), ndl.Tensor(y, device=model.device)
        
        out = model(X)
        loss = loss_fn(out, y)        

        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        total_loss += loss.data.numpy() * y.shape[0]
        total_samples += y.shape[0]
        
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    return correct/total_samples, total_loss/total_samples
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn_instance = loss_fn()
    
    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn_instance, opt)
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn_instance = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn_instance, None)
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    correct = 0
    total_loss = 0.0
    total_samples = 0
    h = None

    # Iterate over the sequence data in steps of `seq_len`, but allow the last
    # chunk to be shorter so that we cover the entire sequence up to the final
    # usable time step (nbatch - 1).
    nbatch = data.shape[0]
    for i in range(0, nbatch - 1, seq_len):
        X, y = ndl.data.ptb_dataset.get_batch(data, i, seq_len, device=device, dtype=dtype)

        out, h = model(X, h)

        # Detach hidden state between truncated BPTT segments
        if isinstance(h, tuple):
            h = tuple([h_i.detach() for h_i in h])
        else:
            h = h.detach() if h is not None else None

        loss = loss_fn(out, y)

        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        total_loss += loss.numpy().item() * y.shape[0]
        total_samples += y.shape[0]

        if opt is not None:
            opt.reset_grad()
            loss.backward()

            if clip is not None:
                opt.clip_grad_norm(clip)

            opt.step()

    avg_acc = correct / total_samples
    avg_loss = total_loss / total_samples
    return float(avg_acc), float(avg_loss)
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn_instance = loss_fn()
    
    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn_instance, opt, clip, device, dtype)
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn_instance = loss_fn()
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn_instance, None, None, device, dtype)
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
