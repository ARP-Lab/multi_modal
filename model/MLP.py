import torch
from torch import nn

from model.UnivarsalNN import UniversalNN


class MLP(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str="",
        input_length: int=0,
        input_width: int=0,
        act_func: str="gelu",
        last_act_off: bool=False
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.input_length = self.local_conf.input_length if self.local_conf.input_length != None else input_length
        self.input_width = self.local_conf.input_width if self.local_conf.input_width != None else input_width
        self.last_act_off = self.local_conf.last_act_off if self.local_conf.last_act_off != None else last_act_off
        
        _act_func = {
            "gelu" : [nn.GELU()],
            "relu" : [nn.Dropout(), nn.ReLU()]
        }
        
        self.backbone_seq = [
            nn.Flatten(),
            nn.Linear(self.input_length * self.input_width, self.local_conf["dim_val"]["1"]),
        ]
        self.backbone_seq += _act_func[act_func]
        
        _dim = [(k, v) for k, v in self._conf.dim_val]
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
            last_act_off: bool=False
        ):
            
            super().__init__()
            
            _act_func = {
                "gelu" : [nn.GELU()],
                "relu" : [nn.Dropout(), nn.ReLU()]
            }
            
            self.backbone_seq = [
                nn.BatchNorm1d(fl_val),
                nn.Linear(fl_val, sl_val),
            ]
            
            if not last_act_off:
                self.backbone_seq += _act_func[act_func]
            
            
        def forward(self, x):
            y = x
            
            for l in self.backbone_seq:
                y = l(y)
                
            return y