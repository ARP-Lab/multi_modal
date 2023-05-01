import os
from dotenv import load_dotenv

_typ = ""
if "uni_nn_type" in os.environ:
    _typ = os.environ["uni_nn_type"]
    if _typ == "keras":
        from tensorflow.keras.layers import layer as unn
    elif _typ == "torch":
        from torch.nn import Module as unn
else:
    raise AssertionError("\"uni_nn_type\" not found") 
        
from zconf.zconf import zconf


class UniversalNN(unn, zconf):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ) -> None:
        
        if _typ == "keras":
            super(UniversalNN, self).__init__()
        elif _typ == "torch":
            unn.__init__()
        
        zconf.__init__(zconf_path, zconf_id)