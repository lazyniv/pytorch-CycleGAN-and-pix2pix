from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np


class BaseTransformer(ABC):
    """
    Base transformer class
    """

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def forward_transform(self, pixel_array: np.ndarray) -> Tensor: ...

    @abstractmethod
    def backward_transform(self, tensor: Tensor) -> np.ndarray: ...

