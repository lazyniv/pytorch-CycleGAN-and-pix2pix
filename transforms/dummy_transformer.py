import torchvision.transforms as transforms
import torch
import numpy as np


from .base_transformer import BaseTransformer
from . import fill_negative_values


class DummyTransformer(BaseTransformer):
    """
    Dummy transformer replaces negative pixel values with 0 and scale the image like (image / opt.scale_factor)
    """

    def __init__(self, opt):
        super().__init__(opt)

    def forward_transform(self, pixel_array: np.ndarray) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.opt.load_size, self.opt.load_size)),
            transforms.Lambda(lambda tensor: fill_negative_values(tensor)),
            transforms.Lambda(lambda tensor: tensor / self.opt.scale_factor),

        ])
        return transform(pixel_array)

    def backward_transform(self, tensor: torch.Tensor) -> np.ndarray:
        image_tensor = tensor.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy *= self.opt.scale_factor
        return image_numpy.astype(np.int32)[0]
