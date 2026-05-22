from turtle import forward
from .tensor import NDArray, Tensor
from typing import Tuple
import numpy as np


class Function:
    def __call__(self, *inputs):
        requires_grad = any(t.requires_grad for t in inputs)
        inputs_data = [t.data for t in inputs]
        output = self.forward(*inputs_data)

        output_tensor = Tensor(output, requires_grad=requires_grad)

        if requires_grad:
            output_tensor._op = self
            output_tensor._inputs = inputs
        
        return output_tensor
    
    def forward(self, *args):
        raise NotImplementedError()
    
    def backward(self, *args):
        pass



class Add(Function):
    def forward(self, a, b):
        return a+b

    def backward(self, out_grad, node):
        return out_grad, out_grad

def add(a, b):
    return Add()(a, b) # __call__

class Mul(Function):
    def forward(self, a, b):
        return a*b
    
    def backward(self, out_grad, node):
        a, b = node._inputs
        return out_grad*b, out_grad*a

def mul(a, b):
    return Mul()(a, b)


class Sub(Function):
    def forward(self, a, b):
        return a-b
    
def sub(a, b):
    return Sub()(a, b)

class Div(Function):
    def forward(self, a, b):
        return a/b
    
def div(a, b):
    return Div()(a, b)

class Pow(Function):
    def forward(self, a, b):
        return a**b

def pow(a, b):
    return Pow()(a, b)

class Transpose(Function):
    def forward(self, a):
        return np.transpose(a)

def transpose(a):
    return Transpose()(a)

class Reshape(Function):
    def forward(self, a, shape):
        return np.reshape(a, shape)

def reshape(a, shape):
    return Reshape()(a, shape)

class Broadcast_To(Function):
    def forward(self, a, shape):
        return np.broadcast_to(a, shape)

def broadcast_to(a, shape):
    return Broadcast_To()(a, shape)

class Summation(Function):
    def forward(self, a):
        return np.sum(a)

def summation(a):
    return Summation()(a)

class MatMul(Function):
    def forward(self, a, b):
        return np.matmul(a, b)

def matmul(a, b):
    return MatMul()(a, b)

class Negate(self, a):
    def forward(self, a):
        return np.negative(a)

def negate(a):
    return Negate()(a)

class Log(Function):
    def forward(self, a):
        return np.log(a)

def log(a):
    return Log()(a)

class Exp(Function):
    def forward(self, a):
        return np.exp(a)

def exp(a):
    return Exp()(a)

class Relu(Function):
    def forward(self, a):
        return np.maximum(0, a)

def relu(a):
    return Relu()(a)

class Sigmoid(Function):
    def forward(self, a):
        return 1 / (1 + np.exp(-a))

def sigmoid(a):
    return Sigmoid()(a)

class Tanh(Function):
    def forward(self, a):
        return np.tanh(a)

def tanh(a):
    return Tanh()(a)

class Sqrt(Function):
    def forward(self, a):
        return np.sqrt(a)

def sqrt(a):
    return Sqrt()(a)

class Abs(Function):
    def forward(self, a):
        return np.abs(a)

def abs(a):
    return Abs()(a)