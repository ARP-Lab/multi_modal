from model.UnivarsalNN import UniversalNN

class weighted_MSELoss(UniversalNN):
    def __init__(
        self,
        weight
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.weight = weight.to(self.glob_conf["device"])
        
    def forward(
        self,
        inputs,
        targets
    ):
        
        return ((inputs - targets) ** 2) * self.weight