import re

from typing import List

import pandas as pd

import torch

from transformers import (
    AutoProcessor, Data2VecAudioModel,
    AutoTokenizer, AutoModel
)

from datasets import (
    Audio, Dataset
)

from utils.dfl import dfl_tools
import omegaconf

import pickle

class AudioTextProcessor(object):
    def __init__(
        self,
        conf_path: str="",
    ) -> None:
        
        _p = dfl_tools.find_dfl_path(
            conf_path, [f"Conf_{self.__class__.__name__}", ".yaml"],
            mode="f", cond="a", only_leaf=True)
        
        assert len(_p) != 0, "File not found."
        
        self._conf = self._read_conf(_p[0])
        
        _txt_model_name = self._conf.pre_trained.text_model
        _wav_model_name = self._conf.pre_trained.audio_model
        
        self._wav_processor = AutoProcessor.from_pretrained(_wav_model_name).cuda() \
            if self._conf.device == "cuda" else AutoProcessor.from_pretrained(_wav_model_name)
        self._txt_tokenizer = AutoTokenizer.from_pretrained(_txt_model_name).cuda() \
            if self._conf.device == "cuda" else AutoTokenizer.from_pretrained(_txt_model_name)
        
        self._wav_model = Data2VecAudioModel.from_pretrained(_wav_model_name).cuda() \
            if self._conf.device == "cuda" else Data2VecAudioModel.from_pretrained(_wav_model_name)
        self._txt_model = AutoModel.from_pretrained(_txt_model_name).cuda() \
            if self._conf.device == "cuda" else AutoModel.from_pretrained(_txt_model_name)
        
        
    def _read_conf(
        self,
        path: str=""
    ):
        
        assert path != "", "Conf file not found."
        
        return omegaconf.OmegaConf.load(path)
    

    def _wav_embedding(
        self,
        wav_files: List[str]=[""]
    )-> List[any]:
        
        _ds = Dataset.from_dict({"wav": wav_files}).cast_column("wav", Audio())
        res = []
        
        for i in range(len(_ds)):
            _sr = _ds[i]['wav']['sampling_rate']
            
            _inputs = self._wav_processor(
                _ds[i]['wav']['array'],
                sampling_rate=_sr,
                return_attention_mask=self._conf.return_attention_mask,
                return_tensors=self._conf.return_tensors,
                padding=self._conf.padding,
                max_length=self._conf.max_length,
                truncation=self._conf.truncation
            )
        
            with torch.no_grad():
                output = self._wav_model(**_inputs)
            
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
                
        _inputs = self._txt_tokenizer(
            _s,
            return_tensors=self._conf.return_tensors,
            padding=self._conf.padding,
            max_length=self._conf.max_length,
            truncation=self._conf.truncation
        )
        
        with torch.no_grad():
            res, _ = self._txt_model(**_inputs, return_dict=False)
        
        return res
    
        
    def make_data(
        self
    ):
        
        anno_path = dfl_tools.find_dfl_path(
            self._conf.target_data_path, ["_eval", ".csv"],
            mode="f", cond="a",
            recur=True, only_leaf=True, res_all=True)
        
        for l in anno_path:
            _all_p, _l_name = l
            _s_anno = _l_name.strip("_eval.csv")
            anno = pd.read_csv(_all_p, skiprows=1)
            f_names = anno[" .1"]
            
            txt_files = [self._conf.target_data_path + "/" + _s_anno + "/"+ f + ".txt" for f in f_names]
            wav_files = [self._conf.target_data_path + "/" + _s_anno + "/"+ f + ".wav" for f in f_names]
            
            txt_embeddings = self._txt_embedding(txt_files=txt_files)
            wav_embeddings = self._wav_embedding(wav_files=wav_files)
            
            data = {
                "file_names": f_names, 
                "txt_embeddings": txt_embeddings, 
                "wav_embeddings": wav_embeddings,
                "EDA": eda,
                "Temp": temp,
                "Emotion": anno.Emotion,
                "Arousal": anno.Arousal,
                "Valence": anno.Valence
            }
            
            if self._conf.device == "cuda":
                torch.cuda.empty_cache()
            
            with open(f"{self._conf.embedding_data_path}/{_s_anno}.pkl", "wb") as f:
                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            

if __name__ == "__main__" :
    ATP = AudioTextProcessor("./model/conf")
    ATP.make_data()