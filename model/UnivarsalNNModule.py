from typing import Tuple, Union

from torch import nn

import omegaconf
from utils.dfl import dfl_tools


class UniversalNN(nn.Module):
    def _glob_conf(
        self,
        func: callable
    ) -> None:
        
        def wrapper(*args, **kwargs):
            _glob_conf_name = f"Conf_glob"
            self._read_conf
            
            res = func(*args, **kwargs)
            
            return res
                
        return wrapper
    
    
    @_glob_conf
    def __init__(
        self,
        conf_path: str="",
        conf_id: str=""
    ) -> None:
        
        super().__init__()
        
        _conf_name = f"Conf_{self.__class__.__name__}" + f"_{conf_id}" if conf_id != "" else ""
        
        _p = dfl_tools.find_dfl_path(
            conf_path, [_conf_name, ".yaml"],
            mode="f", cond="a", only_leaf=True)
        
        assert len(_p) != 0, "File not found."
        
        self._conf = self._read_conf(_p[0])
        
    
    def _read_conf(
        self,
        path: str=""
    ) -> Tuple[Union[omegaconf.DictConfig, omegaconf.ListConfig]]:
        
        assert path != "", "Conf file not found."
        
        return omegaconf.OmegaConf.load(path) 