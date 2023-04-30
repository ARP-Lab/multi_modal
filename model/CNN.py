import torch
from torch import nn

from model.UnivarsalNN import UniversalNN

class ConvNetwork_AudioText(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.backbone_seq = [
            nn.Conv1d(in_channels = 1, out_channels= 10, kernel_size = 15),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.25),
            nn.Conv1d(in_channels = 10, out_channels = 1, kernel_size = 15),
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
        

class ConvNetwork_TimeSeries(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.backbone_seq = [
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=11),
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


class ConvNetwork_TensorFusionMixer(UniversalNN):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.backbone_seq = [
            nn.Conv2d(in_channels = 12, out_channels = 64, kernel_size=3),
            n.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1152, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.25),
            nn.Linear(64, 7)
        ]


    def forward(
        self,
        x
    ):
        
        y = x
        
        for l in self.backbone_seq:
            y = l(y)
            
        return y