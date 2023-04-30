import torch
from torch import nn

from model.UnivarsalNN import UniversalNN
from model.MLP import MLP


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
        self.ModelMLP_fin = MLP(
            conf_path=zconf_path,
            input_length=self.local_conf["mlp_input_length"],
            input_width=self.local_conf["mlp_input_width"]
        ).to(self.glob_conf["device"])
        self.softmax = nn.Softmax(dim=self.local_conf["softmax_dim"])
        
    
    def _tensor_fusion(
        self,
        batch_arr1,
        batch_arr2
    ) -> torch.Tensor:
        
        _fml = []
    
        for i, (arr1, arr2, arr3) in enumerate(zip(batch_arr1, batch_arr2, batch_arr3)):
            
            arr1 = arr1.unsqueeze(0).unsqueeze(0)
            arr2 = arr2.unsqueeze(0).unsqueeze(-1)
            arr3 = arr3.unsqueeze(-1).unsqueeze(-1)

            _km = torch.kron(arr3, torch.kron(arr2, arr1,))
            l, w, d = _km.shape

            _km = _km.view(-1, l, w, d)
            _fml.append(_km)

        _fm = torch.concat(_fml)

        return fusion__fmmatrix
    
        
    def forward(
        self,
        x1,
        x2
    ):
        y1 = self.ModelA(x1)
        y2 = self.ModelB(x2)
        
        y = self._tensor_fusion(y1, y2) 
        
        y = self.ModelMLP_fin(y)
        y = self.softmax(y)
        
        return y     