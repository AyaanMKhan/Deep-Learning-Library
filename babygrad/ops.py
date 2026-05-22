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