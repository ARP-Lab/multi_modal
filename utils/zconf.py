from typing import Tuple, Union
import os

from torch import nn

import omegaconf
from utils.dfl import dfl_tools


class zconf(object):
    def __init__(
        self,
        conf_path: str="",
        conf_id: str=""
    ) -> None:
        
        _glob_os_env_path = os.environ.get("zconf_path")
        
        self.conf_path = _glob_os_env_path if _glob_os_env_path is not None else conf_path
        self.conf = None
        
        assert self.conf_path != "", "zConf path not found"
        
        self._local_conf(conf_id=conf_id)
        
        
    def _get_conf(
        self,
        conf_name: str=""
    ) -> str:
        
        _p = dfl_tools.find_dfl_path(
            self.conf_path, [conf_name, ".yaml"],
            mode="f", cond="a", only_leaf=True)
        
        assert len(_p) != 0, "File not found."
        
        return _p[0]
        
        
    def _glob_conf(
        self,
        func: callable
    ) -> None:
        
        def wrapper(*args, **kwargs):
            _glob_conf_name = f"Conf_glob"
            
            _p = self._get_conf(conf_name=_glob_conf_name)
            self.conf += self._read_conf(_p)
            
            res = func(*args, **kwargs)
            
            return res
                
        return wrapper
    
    
    @_glob_conf
    def _local_conf(
        self,
        conf_id: str=""
    ):
        _conf_name = f"Conf_{self.__class__.__name__}" + f"_{conf_id}" if conf_id != "" else ""
        
        _p = self._get_conf(conf_name=_conf_name)
        self.conf += self._read_conf(_p)
        
    
    def _read_conf(
        self,
        path: str=""
    ) -> Tuple[Union[omegaconf.DictConfig, omegaconf.ListConfig]]:
        
        assert path != "", "Conf file not found."
        
        return omegaconf.OmegaConf.load(path) 