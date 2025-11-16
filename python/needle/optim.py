"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            
            # cast hyperparameters to match parameter dtype
            # lr = np.array(self.lr, dtype=p.data.dtype)
            # momentum = np.array(self.momentum, dtype=p.data.dtype)
            # weight_decay = np.array(self.weight_decay, dtype=p.data.dtype)
            
            grad = p.grad.data
            
            # apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            
            # initialize momentum  
            if p not in self.u:
                self.u[p] = ndl.init.zeros(*p.data.shape, dtype=p.dtype, device=p.device, requires_grad=False)
            
            if self.momentum > 0:
                self.u[p].data = self.momentum * self.u[p].data + (1 - self.momentum) * grad
            else:
                # When momentum is 0, just use the gradient directly
                self.u[p].data = grad
                
            # update params
            p.data = p.data - self.lr * self.u[p].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        total_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                param_norm = np.linalg.norm(p.grad.numpy())
                total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)

        clip_coefficient = max_norm / (total_norm + 1e-6)
        if clip_coefficient < 1:
            for p in self.params:
                if p.grad is not None:
                    p.grad.data = p.grad.data * clip_coefficient
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        
        for p in self.params:
            if p.grad is None:
                continue
            
            # cast hyperparameters to match parameter dtype
            lr = np.array(self.lr, dtype=p.cached_data.dtype)
            weight_decay = np.array(self.weight_decay, dtype=p.cached_data.dtype)
            beta1 = np.array(self.beta1, dtype=p.cached_data.dtype)
            beta2 = np.array(self.beta2, dtype=p.cached_data.dtype)
            eps = np.array(self.eps, dtype=p.cached_data.dtype)
            
            # cast gradient to match parameter dtype
            grad = p.grad.numpy().astype(p.cached_data.dtype)
            
            # apply weight decay
            if self.weight_decay > 0:
                grad = grad + weight_decay * p.cached_data.numpy()
            
            # initialize first and second moment
            if p not in self.m:
                self.m[p] = ndl.Tensor(np.zeros_like(p.cached_data), dtype=p.cached_data.dtype, device=p.device)
            if p not in self.v:
                self.v[p] = ndl.Tensor(np.zeros_like(p.cached_data), dtype=p.cached_data.dtype, device=p.device)
            
            # update first moment estimate
            m_new = (beta1 * self.m[p].numpy() + (1 - beta1) * grad).astype(p.cached_data.dtype)
            self.m[p] = ndl.Tensor(m_new, device=p.device)
            
            # update second moment estimate  
            v_new = (beta2 * self.v[p].numpy() + (1 - beta2) * (grad * grad)).astype(p.cached_data.dtype)
            self.v[p] = ndl.Tensor(v_new, device=p.device)
            
            # bias correction
            beta1_correction = 1 - beta1 ** self.t
            beta2_correction = 1 - beta2 ** self.t 
            m_hat = self.m[p].numpy() / beta1_correction
            v_hat = self.v[p].numpy() / beta2_correction
            
            # update parameters
            update = (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(p.cached_data.dtype)
            update = ndl.Tensor(update, device=p.device, requires_grad=False)
            p.data = p.data - update
        ### END YOUR SOLUTION
