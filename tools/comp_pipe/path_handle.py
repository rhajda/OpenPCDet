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
paths.root = (Path(__file__).parent / '../..').resolve()
paths.tools = paths.root / 'tools'
# paths related to data:
#   input:
paths.indy_main = paths.root / 'data/indy/for_training'

paths.indy_no_noise = paths.indy_main / 'sim_no_noise'
paths.indy_real = paths.indy_main / 'real'
paths.indy_real_local_part = paths.root / 'data_local/indy_up_to_50/real'

#   output:
paths.pipe_results = paths.root / 'comp_pipe/results'

paths.indy_exp_real = paths.indy_real / 'training/velodyne/000009.pcd'
paths.indy_exp_no_noise = paths.indy_no_noise / 'training/velodyne/000009.pcd'

# paths related to configs:
paths.cfg = paths.tools / 'cfgs'
paths.cfg_indy_pointrcnn = paths.cfg / 'indy_models/pointrcnn.yaml'

# Target for results:
# Result is save to /home/output/cfgs/indy_models/pointrcnn/default/eval/eval_with_train/# epoch_0/val