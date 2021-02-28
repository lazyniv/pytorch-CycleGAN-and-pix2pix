import torch.utils.data as data
from abc import ABC, abstractmethod
from transforms import BaseTransformer


class BaseTrainDataset(data.Dataset, ABC):
    def __init__(self, opt, transformer: BaseTransformer):
        self.opt = opt
        self.transformer = transformer
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __getitem__(self, index): ...

    @abstractmethod
    def shuffle_index(self): ...


class BaseTestDataset(data.Dataset, ABC):
    def __init__(self, opt, transformer: BaseTransformer):
        self.opt = opt
        self.transformer = transformer
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __getitem__(self, index): ...

