import os
import sys
import re

from inspect import getouterframes, currentframe

from typing import List, Tuple

from enum import Enum


class dfl_base(object):
    @classmethod
    def exists(
        cls,
        target: str
    ) -> bool:
        
        return os.path.exists(target)
    
    
    @classmethod
    def make_dir(
        cls,
        base_path: str,
        target: str
    ) -> bool:
        
        try:
            _t = f"{base_path}/{target}"
            if not cls.exists(_t):
                os.makedirs(_t)
                return True
            
        except OSError:
            pass
        
        return False
    

class dfl_tools(object):
    @staticmethod
    def _recu_limit_chk(
        func: callable
    ) -> None:
        
        _def_val = 1000 # default of recursive value
        lv = len(getouterframes(currentframe()))
        
        def wrapper(*args, **kwargs):
            if lv == 1:
                sys.setrecursionlimit(_def_val * 10)
                
            res = func(*args, **kwargs)
            
            if lv == 1:
                sys.setrecursionlimit(_def_val)
            
            return res
                
        return wrapper
    
    
    @staticmethod
    def _conv_mode(
        mode: str
    ) -> int:
        
        assert mode != "", "Must input with a mode key."
        
        mode_sep = list(mode)
        res = 0
        
        for m in mode_sep:
            if m == "d":
                res += 0b1 << 0
            elif m == "f":
                res += 0b1 << 1
            elif m == "l":
                res += 0b1 << 2
                
        return res
    
    
    @staticmethod
    def _check_dfl(
        mode_key: str
    ) -> callable:
        
        assert mode_key != "", "Must input with a mode key."
        
        mode_sel_func = {
            f"{dfl_tools.mode_type.d.value}": os.path.isdir,
            f"{dfl_tools.mode_type.f.value}": os.path.isfile,
            f"{dfl_tools.mode_type.l.value}": os.path.islink
        }
        
        return mode_sel_func[f"{mode_key}"]
    

    @classmethod
    def get_dfl_list(
        cls,
        target_path: str,
        mode: str="d"
    ) -> List[Tuple[str, str]]:
        
        assert target_path != "", "Must input a Target path."
        
        res = []
        
        if os.path.exists(target_path):
            all_list = sorted(os.listdir(target_path))

            m = cls._conv_mode(mode)
            
            for l in all_list:
                for i in range(len(cls.mode_type)):
                    x = (m & (2 ** i))
                    
                    if x and cls._check_dfl(f"{x}")(target_path + "/" + l):
                        res.append((target_path + "/" + l, l))
                        break
                    
        return res
    
    
    @classmethod
    def find_key_in_str(
        cls,
        s: str,
        regex_keys: List[str]=[""],
        cond: str="a"
    ):
        
        assert s != "", "Must input a string."
        assert cond != "", "\'cond\' value must include one of the following type alphabet. \'a(and), o(or)\'"
        
        cond = list(cond)
        cond.sort()
        cond = "".join(str(x) for x in cond)
        # and not len(cond) < 2, \
        assert cond != "ao" \
            and (cond != "a" or cond != "o"), "\'cond\' value must include one of the following type only one alphabet."
        
        _founded = 1 if cond == "a" else 0
        
        for regex_key in regex_keys:
            r = re.compile(regex_key)
            _f = 1 if len(r.findall(s)) > 0 else 0
            
            _founded = _founded & _f if cond == "a" else _founded | _f
            
            # if cond == "a":
            #     _founded &= _f
            # elif cond == "o":
            #     _founded |= _f
        
        return True if _founded == 1 else False
    

    @classmethod
    @_recu_limit_chk
    def find_dfl_path(
        cls,
        target_path: str,
        regex_keys: List[str]=[""],
        mode: str="d",
        cond: str="a",
        recur: bool=False,
        only_leaf: bool=False,
        res_all: bool=False
    ) -> List[str]:
        
        assert target_path != "", "Must input a Target path."
        # assert not type(regex_keys) is type(list)
        assert mode != "", "\'mode\' value must include one of the following type strings. \"d(dir), f(file), l(link)\""
        
        _chk_d_opt = False
        _m = list(mode)
        
        if recur and "d" not in _m:
            _m.append("d")
        else:
            _chk_d_opt = True
            
        _m.sort()
        _m = "".join(str(x) for x in _m)
        
        res = []
        
        dfl_list = cls.get_dfl_list(target_path=target_path, mode=_m)
        
        if len(dfl_list):
            for dfl in dfl_list:
                _chk_recu_dir = False
                
                # check a path and directory if exist..
                if recur and cls._check_dfl(f"{cls.mode_type.d.value}")(dfl[0]):
                    res += cls.find_dfl_path(
                        target_path=f"{dfl[0]}",
                        regex_keys=regex_keys,
                        mode=mode,
                        cond=cond,
                        recur=recur,
                        only_leaf=only_leaf,
                        res_all=only_leaf
                    )
                    
                    if not _chk_d_opt:
                        _chk_recu_dir = True
                
                _s = dfl[1] if only_leaf else dfl[0]
                
                if not _chk_recu_dir and cls.find_key_in_str(s=_s, regex_keys=regex_keys, cond=cond):
                    res.append(dfl if res_all else dfl[0])
        
        return res
    
    
    class mode_type(Enum):
        d = 0b1 << 0
        f = 0b1 << 1
        l = 0b1 << 2