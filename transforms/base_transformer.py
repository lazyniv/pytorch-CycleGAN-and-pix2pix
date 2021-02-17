from abc import ABC, abstractmethod
import torchvision.transforms as transforms


class BaseTransformer(ABC):
    """
    Base transformer class
    """

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def get_transform(self) -> transforms.Compose: ...

    @abstractmethod
    def get_reverse_transform(self) -> transforms.Compose: ...

