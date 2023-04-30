from torch import nn
from utils.zconf import zconf


class UniversalNN(nn.Module, zconf):
    def __init__(
        self,
        conf_path: str="",
        conf_id: str=""
    ) -> None:
        
        nn.Module().__init__()
        zconf().__init__(conf_path, conf_id)