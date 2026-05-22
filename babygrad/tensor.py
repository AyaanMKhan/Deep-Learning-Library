from typing import List
import numpy as np
from numpy._typing import NDArray
NDArray = np.ndarray

class Tensor:
    def __init__(self, data, *, device=None, dtype="float32", requires_grad=True):
        self.data = data
        self.device = device
        self.dtype= dtype
        self.requires_grad = requires_grad


        if isinstance(self.data, Tensor):
            pass

        if isinstance(self.data, NDArray):
            pass

        if isinstance(self.data, List):
            pass

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        from .ops import Add
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Add()(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        from .ops import Mul
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul()(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        from .ops import Pow
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Pow()(self, other)
    
    def __rpow__(self, other):
        return self.__pow__(other)
    
    def __sub__(self, other):
        from .ops import Sub
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(self, other)
    
    
    def __truediv__(self, other):
        from .ops import Div
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Div()(self, other)
    
    
    def __matmul__(self, other):
        from .ops import MatMul
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return MatMul()(self, other)
    
    def __rmatmul__(self, other):
        return self.__matmul__(other)
    
    def sum(self):
        from .ops import Summation
        return Summation()(self)
    
    def broadcast_to(self, shape):
        from .ops import Broadcast_To
        return Broadcast_To()(self, shape)
    
    def reshape(self, shape):
        from .ops import Reshape
        return Reshape()(self, shape)
    
    def transpose(self):
        from .ops import Transpose
        return Transpose()(self)
    

    def matmul(self, other):
        from .ops import MatMul
        return MatMul()(self, other)
    
    def __neg__(self):
        from .ops import Negate
        return Negate()(self)
    
    
    
    


    def backward(self, out_grad=None):
        pass

    
    def numpy(self):
        return self.data.copy()
    
    def detach(self):
        return Tensor(self.data, requires_grad=False)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def device(self):
        return self._device
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    