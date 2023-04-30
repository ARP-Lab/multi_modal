from torch import nn
from utils.zconf import zconf


class UniversalNN(nn.Module, zconf):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ) -> None:
        
        nn.Module.__init__()
        zconf.__init__(zconf_path, zconf_id)