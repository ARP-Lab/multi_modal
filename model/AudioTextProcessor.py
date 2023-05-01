from typing import List
import re

import pandas as pd

import torch

from transformers import (
    AutoProcessor, Data2VecAudioModel,
    AutoTokenizer, AutoModel
)

from datasets import (
    Audio, Dataset
)

from zconf.zconf import zconf
from dfl.dfl import dfl_base, dfl_tools

import pickle
import csv


class AudioTextProcessor(zconf):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ) -> None:
        
        super().__init__(zconf_path=zconf_path, zconf_id=zconf_id)
        
        _txt_model_name = self.local_conf["pre_trained"]["text_model"]
        _wav_model_name = self.local_conf["pre_trained"]["audio_model"]
        
        self.processor = {
            "wav": AutoProcessor.from_pretrained(_wav_model_name),
            "txt": AutoTokenizer.from_pretrained(_txt_model_name)
        }
        self.model = {
            "wav": Data2VecAudioModel.from_pretrained(_wav_model_name),
            "txt": AutoModel.from_pretrained(_txt_model_name)
        }
        
        if self.glob_conf.device == "cuda":
            for _i in ["wav", "txt"]:
                # self.processor[_i] = self.processor[_i].cuda()
                self.model[_i] = self.model[_i].cuda()
                

    def _wav_embedding(
        self,
        wav_files: List[str]=[""]
    )-> List[any]:
        
        _ds = Dataset.from_dict({"wav": wav_files}).cast_column("wav", Audio())
        res = []
        
        for i in range(len(_ds)):
            _sr = _ds[i]['wav']['sampling_rate']
            
            _inputs = self.processor["wav"](
                _ds[i]['wav']['array'],
                sampling_rate=_sr,
                return_attention_mask=self.local_conf["audio_conf"]["return_attention_mask"],
                return_tensors=self.local_conf["audio_conf"]["return_tensors"],
                padding=self.local_conf["audio_conf"]["padding"],
                max_length=self.local_conf["audio_conf"]["max_length"],
                truncation=self.local_conf["audio_conf"]["truncation"]
            )
        
            with torch.no_grad():
                output = self.model["wav"](**_inputs)
            
            res.append(output['last_hidden_state'])
        
        return res
    
    
    def _txt_embedding(
        self,
        txt_files: List[str]=[""]
    ) -> List[any]:
        
        _s = []

        for tl in txt_files:
            with open(tl, "r") as f:
                _l = f.readline()
                _l = re.sub('\n', '', _l)
                _l = re.sub('  ', ' ', _l)
                _l = _l.rstrip().lstrip()
                _s.append(_l)
                
        _inputs = self.processor["txt"](
            _s,
            return_tensors=self.local_conf["text_conf"]["return_tensors"],
            padding=self.local_conf["text_conf"]["padding"],
            max_length=self.local_conf["text_conf"]["max_length"],
            truncation=self.local_conf["text_conf"]["truncation"]
        )
        
        with torch.no_grad():
            res, _ = self._txt_model(**_inputs, return_dict=False)
        
        return res
    
        
    def make_data(
        self,
        year: int=None
    ) -> pd.DataFrame:
        
        assert year is not None, "\"year\" param must input."
        assert not (year < 19 and year > 20)
        
        path = self.glob_conf["org_data_path"][year]
        assert path is not ("" or None)
        
        # if dfl_base.exists(f"{self._conf.embedding_data_path}"):
        #     dfl_base.make_dir(f"{self._conf.embedding_data_path}")
                
        anno_path = dfl_tools.find_dfl_path(
            path, ["_eval", ".csv"],
            mode="f", cond="a",
            recur=True, only_leaf=True, res_all=True)
        
        data = {}
        
        for _a in anno_path:
            _all_p, _l_name = _a
            _s_anno = _l_name.strip("_eval.csv")
            anno = pd.read_csv(_all_p, skiprows=1)
            f_names = anno[" .1"]
            
            txt_files = [self.glob_conf["data_path"] + "/" + _s_anno + "/"+ f + ".txt" for f in f_names]
            wav_files = [self.glob_conf["data_path"] + "/" + _s_anno + "/"+ f + ".wav" for f in f_names]
            
            _txt_embed = self._txt_embedding(txt_files=txt_files)
            _wav_embed = self._wav_embedding(wav_files=wav_files)
            
            _td = {
                "file_names": f_names, 
                "txt_embeddings": _txt_embed, 
                "wav_embeddings": _wav_embed,
                "Emotion": anno.Emotion,
                "Arousal": anno.Arousal,
                "Valence": anno.Valence
            }
            
            if self.glob_conf["device"] == "cuda":
                torch.cuda.empty_cache()
            
            _now_model_audio = self.local_conf["pre_trained"]["audio_model"].split("/")
            _now_model_audio = _now_model_audio[0] if len(_now_model_audio) < 2 else _now_model_audio[1]
            _now_model_text = self.local_conf["pre_trained"]["text_model"].split("/")
            _now_model_text = _now_model_text[0] if len(_now_model_text) < 2 else _now_model_text[1]
            
            if self.local_conf["save_pickle"]:
                with open(self.glob_conf["data_path"] + "/pkl" + f"/{_s_anno}_AudioText_{_now_model_audio}_.pkl", "wb") as f:
                    pickle.dump(_td, f, pickle.HIGHEST_PROTOCOL)
                    
            if self.local_conf["save_pickle"]:
                _tdf = pd.DataFrame.from_dict(_td, orient='index')
                _tdf.to_csv(self.glob_conf["data_path"] + "/csv" + f"{_s_anno}_AudioText_{_now_model_audio}.csv", sep=",")
            
            data[_s_anno] = _td
                
        return data