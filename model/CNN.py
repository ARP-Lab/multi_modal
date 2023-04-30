import torch
from torch import nn

from utils.UnivarsalNN import UniversalNN


class CNN_TS_First(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        
        self.backbone_seq = [
            nn.Conv1d(**self.local_conf["conv1d_conf"][1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.local_conf["BatchNorm1d_conf"]),
            nn.Dropout(p=self.local_conf["Dropout_prob"]),
            nn.Conv1d(**self.local_conf["conv1d_conf"][2]),
            nn.ReLU()
        ]


    def forward(
        self,
        x
    ):
        
        y = torch.transpose(x, 1, 2)
        
        for l in self.backbone_seq:
            y = l(y)
            
        return y
        

class CNN_TS_Merge(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.backbone_seq = [
            nn.Conv1d(**self.local_conf["conv1d_conf"]),
            nn.ReLU()
        ]


    def forward(
        self,
        x
    ):
        
        y = x
        
        for l in self.backbone_seq:
            y = l(y)
            
        return y.squeeze()


class CNN_TensorFusionMixer(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.backbone_seq = [
            nn.Conv2d(**self.local_conf["conv1d_conf"][1]),
            n.LeakyReLU(),
            nn.MaxPool2d(self.local_conf["MaxPool2d_conf"]),
            nn.Conv2d(**self.local_conf["conv1d_conf"][2]),
            nn.LeakyReLU(),
            nn.MaxPool2d(self.local_conf["MaxPool2d_conf"]),
            nn.Flatten(),
            nn.Linear(**self.local_conf["Linear_conf"][1]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.local_conf["BatchNorm1d_conf"]),
            nn.Dropout(p=self.local_conf["Dropout_prob"]),
            nn.Linear(**self.local_conf["Linear_conf"][2])
        ]


    def forward(
        self,
        x
    ):
        
        y = x
        
        for l in self.backbone_seq:
            y = l(y)
            
        return y