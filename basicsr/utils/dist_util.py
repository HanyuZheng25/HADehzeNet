# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
这段代码是用于分布式训练的工具函数，主要用于初始化分布式训练环境、获取分布式信息以及限制仅主进程执行某些函数等。

其中包含了以下函数：

init_dist(launcher, backend='nccl', **kwargs): 初始化分布式训练环境的函数。根据不同的 launcher（启动器），调用不同的初始化函数进行分布式训练环境的设置。

_init_dist_pytorch(backend, **kwargs): 使用 PyTorch 原生的分布式训练方式进行初始化的函数。根据环境变量设置每个进程的 GPU 设备，
并调用 torch.distributed.init_process_group() 初始化分布式训练组。

_init_dist_slurm(backend, port=None): 使用 SLURM 环境进行初始化的函数。根据 SLURM 的环境变量设置每个进程的 GPU 设备，并使用 subprocess 
获取主节点的地址，然后通过环境变量设置的方式设置主节点地址和端口，最后调用 torch.distributed.init_process_group() 初始化分布式训练组。

get_dist_info(): 获取当前分布式训练的信息，包括当前进程的 rank 和总的进程数。

master_only(func): 装饰器函数，用于限制仅主进程执行某些函数。在装饰的函数中，获取当前进程的 rank，如果是主进程（rank 为 0），则执行原始函数，
否则不执行。

这些函数提供了方便的工具来管理分布式训练环境，并且能够很容易地在不同的分布式环境中进行切换和配置。
"""

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
