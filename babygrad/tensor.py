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
    