import torch
from torch import nn

from model.MLP import MLP
from utils.UnivarsalNN import UniversalNN


class TensorFusionMixer(UniversalNN):
    def __init__(
        self,
        ModelA: nn.Module,
        ModelB: nn.Module,
        ModelC: nn.Module,
        ModelD: nn.Module,
        ModelE: nn.Module,
        ModelF: nn.Module,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_id=zconf_id)
        
        self.ModelA = ModelA
        self.ModelB = ModelB
        self.ModelC = ModelC
        self.ModelD = ModelD
        self.ModelE = ModelE
        self.Model_cnn_final = ModelF
        self.softmax = nn.Softmax(dim=self.local_conf["softmax_dim"])
        
    
    def _tensor_fusion(
        self,
        batch_arr1,
        batch_arr2,
        batch_arr3
    ) -> torch.Tensor:
        
        _fml = []
    
        for (arr1, arr2, arr3) in zip(batch_arr1, batch_arr2, batch_arr3):
            
            arr1 = arr1.unsqueeze(0).unsqueeze(0)
            arr2 = arr2.unsqueeze(0).unsqueeze(-1)
            arr3 = arr3.unsqueeze(-1).unsqueeze(-1)

            _km = torch.kron(arr3, torch.kron(arr2, arr1,))
            l, w, d = _km.shape

            _km = _km.view(-1, l, w, d)
            _fml.append(_km)

        _fm = torch.concat(_fml)

        return _fm
    
        
    def forward(
        self,
        x1,
        x2,
        x3,
        x4
    ):
        y1 = self.ModelA(x1)
        y2 = self.ModelB(x2)
        y3 = self.ModelA(x3)
        y4 = self.ModelB(x4)
        y5 = torch.concat([y3, y4], dim=1)
        y = self.ModelE(y5)
        
        y = self._tensor_fusion(y1, y2, y5)
        
        y = self.Model_cnn_final(y)
        y = self.softmax(y)
        
        return y     