import torch
from torch import nn

from utils.UnivarsalNN import UniversalNN


class MLP(UniversalNN):
    def __init__(
        self,
        input_length: int=0,
        input_width: int=0,
        act_func: str="gelu",
        last_act_off: bool=False,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.input_length = self.check_local_var("input_length", input_length)
        self.input_width = self.check_local_var("input_width", input_width)
        self.act_func = self.check_local_var("act_func", act_func)
        self.last_act_off = self.check_local_var("act_func", last_act_off)
        
        _act_func = {
            "gelu" : [nn.GELU()],
            "relu" : [nn.Dropout(), nn.ReLU()]
        }
        
        self.backbone_seq = [
            nn.Flatten(),
            nn.Linear(self.input_length * self.input_width, self.local_conf["dim_val"]["1"]),
        ]
        self.backbone_seq += _act_func[self.act_func]
        
        _dim = [(k, v) for k, v in self.local_conf["dim_val"]]
        for i in range(len(_dim) - 1):
            _last_act_off = False
            
            if i == len(_dim) - 1 and self.last_act_off:
                _last_act_off = True
            self.backbone_seq += [
                self.MLP_block(_dim[i][1], _dim[i + 1][1], _last_act_off)
            ]
            
        
    def forward(self, x):
        y = x
        
        for l in self.backbone_seq:
            y = l(y)
            
        return y
    
    
    class MLP_block(UniversalNN):
        def __init__(
            self,
            fl_val: int,
            sl_val: int,
            act_func: str="gelu",
            last_act_off: bool=False,
            zconf_path: str="",
            zconf_id: str=""
        ):
            
            super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
            
            self.fl_val = self.check_local_var("fl_val", fl_val)
            self.sl_val = self.check_local_var("sl_val", sl_val)
            self.act_func = self.check_local_var("act_func", act_func)
            self.last_act_off = self.check_local_var("last_act_off", last_act_off)
            
            _act_func = {
                "gelu" : [nn.GELU()],
                "relu" : [nn.Dropout(), nn.ReLU()]
            }
            
            self.backbone_seq = [
                nn.BatchNorm1d(self.fl_val),
                nn.Linear(self.fl_val, self.sl_val)
            ]
            
            if not last_act_off:
                self.backbone_seq += _act_func[self.act_func]
            
            
        def forward(self, x):
            y = x
            
            for l in self.backbone_seq:
                y = l(y)
                
            return y