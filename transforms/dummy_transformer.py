import torchvision.transforms as transforms


from .base_transformer import BaseTransformer
from . import fill_negative_values


class DummyTransformer(BaseTransformer):
    """
    Dummy transformer replaces negative pixel values with 0 and scale the image like (image / opt.scale_factor)
    """

    def __init__(self, opt):
        super().__init__(opt)

    def get_transform(self) -> transforms.Compose:
        transform = [
            transforms.ToTensor(),
            transforms.Resize((self.opt.load_size, self.opt.load_size)),
            transforms.Lambda(lambda tensor: fill_negative_values(tensor)),
            transforms.Lambda(lambda tensor: tensor / self.opt.scale_factor),

        ]
        return transforms.Compose(transform)

    def get_reverse_transform(self) -> transforms.Compose:
        transform = [
            transforms.Lambda(lambda tensor: tensor * self.opt.scale_factor),
        ]
        return transforms.Compose(transform)
