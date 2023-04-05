from torch import nn

from model.UnivarsalNNModule import UniversalNN


class MLP(UniversalNN):
    def __init__(
        self,
        conf_path: str="",
        conf_id: str="",
        input_length: int=0,
        input_width: int=0,
        last_act_off: bool=False
    ):
        
        super().__init__(conf_path=conf_path, conf_id=conf_id)
        
        self.input_length = self._conf.input_length if self._conf.input_length != None else input_length
        self.input_width = self._conf.input_width if self._conf.input_width != None else input_width
        self.last_act_off = self._conf.last_act_off if self._conf.last_act_off != None else last_act_off
        
        self.backbone_seq = [
            nn.Flatten(),
            nn.Linear(self.input_length * self.input_width, self._conf.dim_val["1"]),
            nn.GELU()
        ]
        
        _dim = [(k, v) for k, v in self._conf.dim_val]
        for i in range(len(_dim) - 1):
            _last_act_off = False
            
            if i == len(_dim) - 1 and self.last_act_off:
                _last_act_off = True
            self.backbone_seq.append(self.MLP_block(_dim[i][1], _dim[i + 1][1], _last_act_off))
            
        
    def forward(self, x):
        y = x
        
        for l in self.backbone_seq:
            y = l(y)
            
        return y
    
    
    class MLP_block(nn.Module):
        def __init__(
            self,
            fl_val: int,
            sl_val: int,
            last_act_off: bool=False
        ):
            
            super().__init__()
            
            self.backbone_seq = [
                nn.BatchNorm1d(fl_val),
                nn.Linear(fl_val, sl_val),
            ]
            
            if not last_act_off:
                self.backbone_seq.append(nn.GELU())
            
            
        def forward(self, x):
            y = x
            
            for l in self.backbone_seq:
                y = l(y)
                
            return y