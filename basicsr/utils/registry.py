# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501

"""
这段代码定义了一个通用的注册表（Registry）类，用于管理对象的注册和获取。

- `Registry` 类具有以下主要方法和属性：
  - `__init__`: 初始化注册表，需要传入注册表的名称。
  - `_do_register`: 内部方法，用于实际注册对象。
  - `register`: 用于注册对象的装饰器或函数。如果作为装饰器使用，则注册对象的名称为其类名或函数名；如果作为函数使用，则需要手动传入对象。
  - `get`: 根据对象名称获取对象。
  - `__contains__`: 判断注册表中是否包含指定名称的对象。
  - `__iter__`: 迭代注册表中的所有对象。
  - `keys`: 返回注册表中所有对象的名称列表。

- 代码中定义了几个具体的注册表实例：
  - `DATASET_REGISTRY`: 数据集注册表，用于注册数据集相关的对象。
  - `ARCH_REGISTRY`: 架构注册表，用于注册模型架构相关的对象。
  - `MODEL_REGISTRY`: 模型注册表，用于注册模型相关的对象。
  - `LOSS_REGISTRY`: 损失函数注册表，用于注册损失函数相关的对象。
  - `METRIC_REGISTRY`: 评价指标注册表，用于注册评价指标相关的对象。

这样的设计模式使得用户可以方便地注册和获取各种对象，例如数据集、模型、损失函数等，从而实现模块化和可扩展的代码结构。
"""
class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        print("register",obj)
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        print(self._obj_map)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
