import torch
import importlib
from .base_transformer import BaseTransformer


def find_transformer_using_name(transformer_name):
    """Import the module "transformer/[transformer_name]_transformer.py".

        In the file, the class called TransformerNameTransformer() will
        be instantiated. It has to be a subclass of BaseTransformer,
        and it is case-insensitive.
        """
    transformer_filename = "transforms." + transformer_name + "_transformer"
    transformer_lib = importlib.import_module(transformer_filename)

    transformer = None
    target_transformer_name = transformer_name.replace('_', '') + 'transformer'
    for name, cls in transformer_lib.__dict__.items():
        if name.lower() == target_transformer_name.lower() and issubclass(cls, BaseTransformer):
            transformer = cls

    if transformer is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseTransformer with class name that matches %s "
            "in lowercase." % (transformer_filename, target_transformer_name)
        )

    return transformer


def fill_negative_values(tensor: torch.Tensor) -> torch.Tensor:
    tensor[tensor < 0] = 0
    return tensor
