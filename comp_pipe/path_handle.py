from pathlib import Path
from easydict import EasyDict
from typing import Any, Union


class Singleton(type):
    """
    Singleton class that can be passed as metaclass to avoid multiple instantiation. 
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class PathSingle(EasyDict, metaclass=Singleton):
    """
    A path singleton class to make it easy to handle valid paths. 
    """

    def __setattr__(self, k: Any, v: Union[Path, str]) -> None:
        v = Path(v) if type(v) == str else v
        assert v.exists
        return super().__setitem__(k, v)
    
    __setitem__ = __setattr__

paths = PathSingle()
paths.root = (Path(__file__).parent / '../').resolve()
paths.indy_main = paths.root / 'data/indy'
paths.indy_no_noise = paths.indy_main / 'sim_no_noise'
paths.indy_real = paths.indy_main / 'real'
