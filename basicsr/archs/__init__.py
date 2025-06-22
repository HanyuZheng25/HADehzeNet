
import importlib

from copy import deepcopy

from os import path as osp

from utils import get_root_logger, scandir

from utils.registry import ARCH_REGISTRY

__all__ = ['build_network']


arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]


_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]

def build_network(opt):
    opt = deepcopy(opt)  # 创建选项的深拷贝,Deepcopy是Python中的一个函数,它可以对Python对象进行深度复制。在进行复制操作时,会将原始对象完全复制一份,并在内存中重新分配一个新的地址。
    network_type = opt.pop('type')  # 从选项中移除 'type' 键，并将其值赋给 network_type
    net = ARCH_REGISTRY.get(network_type)(**opt)  # 根据网络类型和选项创建网络实例
    logger = get_root_logger()  # 获取根记录器
    logger.info(f'网络 [{net.__class__.__name__}] 已创建。')  # 记录网络创建过程
    return net  # 返回创建的网络实例

