from typing import Tuple, Union
import os

from utils.dfl import dfl_tools

import omegaconf
from dotenv import load_dotenv


class zconf(object):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ) -> None:
        
        load_dotenv()
        self.sys_conf = {_i:os.environ[_i] for _i in set(os.environ)}
        self.glob_conf = None
        self.local_conf = None
        
        self.zconf(zconf_path=zconf_path, zconf_id=zconf_id)
        
        
    def __get_path(
        self,
        zconf_name: str=""
    ) -> str:
        
        _p = dfl_tools.find_dfl_path(
            self.zconf_path, [zconf_name, ".yaml"],
            mode="f", cond="a", only_leaf=True)
        
        assert len(_p) != 0, f"zconf File has not found({self.zconf_path}/{zconf_name})."
        
        return _p[0]
        
        
    def __get_glob(
        func: callable
    ) -> None:
        
        def wrapper(self, *args, **kwargs):
            _glob_conf_name = f"zconf_glob"
            
            _p = self.__get_path(zconf_name=_glob_conf_name)
            self.glob_conf = self.__read_zconf(_p)
            
            res = func(self, *args, **kwargs)
            
            return res
                
        return wrapper
    
    
    @__get_glob
    def __get_local(
        self,
        zconf_id: str=""
    ) -> None:
        
        _conf_name = f"zconf_{str(self.__class__.__name__)}"
        _conf_name += f"_{zconf_id}" if zconf_id != "" else ""
        
        _p = self.__get_path(zconf_name=_conf_name)
        self.local_conf = self.__read_zconf(_p)
        
    
    def __read_zconf(
        self,
        path: str=""
    ) -> Tuple[Union[omegaconf.DictConfig, omegaconf.ListConfig]]:
        
        assert path != "", "\"path\" param must input."
        
        return omegaconf.OmegaConf.load(path) 
    
    
    def zconf(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ) -> None:
    
        _zconf_path = self.sys_conf["zconf_path"] if "zconf_path" in self.sys_conf else ""
        self.zconf_path = _zconf_path if _zconf_path is not None else zconf_path
        
        assert self.zconf_path != "", "zconf path not found"
        
        self.__get_local(zconf_id=zconf_id)