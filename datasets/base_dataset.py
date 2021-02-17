import torch.utils.data as data
from abc import ABC, abstractmethod
from transforms import BaseTransformer


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of datasets.
    -- <__getitem__>:                   get a datasets point.
    -- <modify_commandline_options>:    (optionally) add datasets-specific options and set default options.
    """

    def __init__(self, opt, transformer: BaseTransformer):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.transformer = transformer
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new datasets-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the datasets."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a datasets point and its metadata information.

        Parameters:
            index - - a random integer for datasets indexing

        Returns:
            a dictionary of datasets with their names. It ususally contains the datasets itself and its metadata information.
        """
        pass

    @abstractmethod
    def shuffle_index(self):
        """Index Shuffling strategy"""
        pass

