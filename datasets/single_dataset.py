import os
from typing import Dict, Any
from random import shuffle

from datasets.base_dataset import BaseDataset
from datasets.utils import load_dcm_paths_slices
from datasets.utils import load_image
from transforms.base_transformer import BaseTransformer


class SingleDataset(BaseDataset):
    """This datasets class can load a set of images specified by the path --dataroot /path/to/datasets.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt, transformer):
        """Initialize this datasets class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, transformer)
        self.A_paths = sorted(load_dcm_paths_slices(self.dir_A))
        self.transform = transformer.get_transform()
        self.reverse_transform = transformer.get_reverse_transform()

    def __getitem__(self, index):
        """Return a datasets point and its metadata information.

        Parameters:
            index - - a random integer for datasets indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the datasets."""
        return len(self.A_paths)
