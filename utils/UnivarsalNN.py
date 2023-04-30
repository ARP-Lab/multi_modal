import os
from dotenv import load_dotenv

type = ""
if "uni_nn_type" in os.environ:
    type = os.environ["uni_nn_type"]
    if type == "keras":
        from tensorflow.keras.layers import layer as unn
    elif type == "torch":
        from torch.nn import Module as unn
else:
    raise AssertionError("\"uni_nn_type\" not found") 
        
from utils.zconf import zconf


class UniversalNN(unn, zconf):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ) -> None:
        
        if type == "keras":
            super(UniversalNN, self).__init__()
        elif type == "torch":
            unn.__init__()
        
        zconf.__init__(zconf_path, zconf_id)