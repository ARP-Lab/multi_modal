from utils.UnivarsalNN import UniversalNN


class weighted_MSELoss(UniversalNN):
    def __init__(
        self,
        weight,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_id=zconf_id)
        
        self.weight = weight.to(self.glob_conf["device"])
    
    
    def forward(
        self,
        inputs,
        targets
    ):
        
        return ((inputs - targets) ** 2) * self.weight