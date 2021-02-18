import os
from typing import Dict, Any
from random import shuffle

from datasets.base_dataset import BaseDataset
from datasets.utils import load_dcm_paths_slices
from datasets.utils import load_image
from transforms.base_transformer import BaseTransformer


class UnpairedMRIDataset(BaseDataset):

    def __init__(self, opt, transformer: BaseTransformer):
        BaseDataset.__init__(self, opt, transformer)

        self.file_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.file_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(load_dcm_paths_slices(self.file_A))
        self.B_paths = sorted(load_dcm_paths_slices(self.file_B))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_A = transformer.get_transform()
        self.transform_B = transformer.get_transform()

        self.reverse_transform_A = transformer.get_reverse_transform()
        self.reverse_transform_B = transformer.get_reverse_transform()

        if self.A_size < self.B_size:
            self.B_index = list(range(0, self.B_size))
            self.A_index = (list(range(0, self.A_size)) * (self.B_size // self.A_size + 1))[:self.B_size]
        else:
            self.A_index = list(range(0, self.A_size))
            self.B_index = (list(range(0, self.B_size)) * (self.A_size // self.B_size + 1))[:self.A_size]

    def __len__(self) -> int:
        return max(self.A_size, self.B_size)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        A_path = self.A_paths[self.A_index[index]]
        B_path = self.B_paths[self.B_index[index]]

        A_img = load_image(A_path)
        B_img = load_image(B_path)

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path
        }

    def shuffle_index(self):
        shuffle(self.A_index)
        shuffle(self.B_index)





