import importlib
from copy import deepcopy
from os import path as osp

from utils import get_root_logger, scandir
from utils.registry import MODEL_REGISTRY
import sys


__all__ = ['build_model']


model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]

_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]

def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
