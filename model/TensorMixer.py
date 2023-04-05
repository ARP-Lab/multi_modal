import torch
from torch import nn

from model.UnivarsalNNModule import UniversalNN
from model.MLP import MLP


class TensorMixer(UniversalNN):
    def __init__(
        self,
        ModelA: nn.Module,
        ModelB: nn.Module,
        conf_path: str="",
        conf_id: str=""
    ):
        
        super().__init__(conf_path=conf_path, conf_id=conf_id)
        
        self.ModelA = ModelA
        self.ModelB = ModelB
        self.ModelMLP_fin = MLP(
            conf_path=conf_path,
            input_length=self._conf.mlp_input_length,
            input_width=self._conf.mlp_input_width
        ).to(device)
        self.softmax = nn.Softmax(dim=self._conf.softmax_dim)
        
    
    def _tensor_fusion(
        self,
        batch_arr1,
        batch_arr2
    ) -> torch.Tensor:
        
        _fml = []
        
        for arr1, arr2 in zip(batch_arr1, batch_arr2):
            _om = torch.outer(arr1, arr2)
            l, w = _om.shape
            _om = _om.view(1, l, w)
            _fml.append(_om)
            
        _fm = torch.concat(_fml)
        
        return _fm
    
        
    def forward(self, x1, x2):
        y1 = self.ModelA(x1)
        y2 = self.ModelB(x2)
        
        y = self._tensor_fusion(y1, y2) 
        
        y = self.ModelMLP_fin(y)
        y = self.softmax(y)
        
        return y     