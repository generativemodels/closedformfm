

import os
from hydra.core.config_store import ConfigStore
import inspect

def register_structured_conf(Config, name = None):
    if name is None: # get the caller module name, to use as config name
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        filename = module.__file__
        name = os.path.basename(filename).replace(".py", "")

    cs = ConfigStore.instance()
    cs.store(name=name, node=Config)

