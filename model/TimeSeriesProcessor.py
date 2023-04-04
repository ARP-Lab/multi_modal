import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import omegaconf
from utils.dfl import dfl_base, dfl_tools

import pickle


class TimeSeriesProcessor(object):
    def __init__(
        self,
        conf_path: str=""
    ):
        
        _p = dfl_tools.find_dfl_path(
            conf_path, [f"Conf_{self.__class__.__name__}", ".yaml"],
            mode="f", cond="a", only_leaf=True)
        
        assert len(_p) != 0, "File not found."
        
        self._conf = self._read_conf(_p[0])
        
        
    def _read_conf(
        self,
        path: str=""
    ):
        
        assert path != "", "Conf file not found."
        
        return omegaconf.OmegaConf.load(path)
    
    
    def _from_array_to_list(
        self,
        x
    ):
        if type(x) != type(np.array([])):
            return x
        else: 
            return x.tolist()


    def _sequence_difference(
        self,
        ts_list
    ):
        
        if type(ts_list) != type([]):
            return ts_list
        else:
            _ts_df = pd.DataFrame(ts_list)
            res = _ts_df.diff()[1:].values
            res = res.reshape(len(res),) 
                
            return res
        
    
    def _make_data_from_annotation(
        self,
        path: str
    ) -> None:
        
        _ts = pd.read_csv(path)
        
        _ts = _ts[['Segment ID', 'Total Evaluation',' .1',' .2']]
        _ts.columns = ['segment_id','emotion','valence','arousal']
        
        _ts = _ts.drop([0], axis = 0).sort_values('segment_id', ascending=True)
        
        _ts['eda'] = 0
        _ts['temp'] = 0
        _ts['ibi'] = 0
        
        _ts = _ts.astype({'eda':'object', 'temp':'object', 'ibi':'object'}).reset_index(drop=True)
        
        return _ts
    
    def _make_data_from_timeseries(
        self,
        path: str
    ) -> None:
        
        _ts = pd.read_csv(path)
        _ts = _ts.rename(columns={'acc' : 'eda'}).drop(['timestamp'], axis = 1)
        
        _ts = _ts.astype({'eda':'object', 'temp':'object', 'ibi':'object'})
    
        return _ts

        pass
    
    def make_data(
        self
    ) -> None:
        
        if dfl_base.exists(f"{self._conf.embedding_data_path}"):
            dfl_base.make_dir(f"{self._conf.embedding_data_path}")
        
        sess_path = dfl_tools.find_dfl_path(
            self._conf.target_data_path, ["Session", ".csv"],
            mode="f", cond="a",
            recur=True, only_leaf=True, res_all=True)
        
        anno_path = dfl_tools.find_dfl_path(
            self._conf.target_data_path, ["_eval", ".csv"],
            mode="f", cond="a",
            recur=True, only_leaf=True, res_all=True)
        
        for l in sess_path + anno_path:
            _all_p_sess, _l_name_sess, _all_p_anno, _l_name_anno = l
            _s_anno = _l_name_anno.strip("_eval.csv")
            
            _data_anno = self._make_data_from_annotation(path=_all_p_anno) # for annotations
            _si = _data_anno['segment_id']
            _data_sess = self._make_data_from_timeseries(path=_all_p_sess) # for sessions
            
            for _s in range(len(_si)):
                _lx = ["eda", "temp", "ibi"]
                for l in _lx:
                    _d = _data_sess[_data_sess['sid'] == _si[_s]][l].dropna(axis=0).values
                    
                    if len(list(_d)) == 0:
                        _data_anno[l].iloc[_s] = np.NaN
                    else: 
                        _data_anno[l].iloc[_s] = list(_d)
                        
            _sids = list(_si[1:].values)
            for _s in range(len(_sids)):
                _tr = _data_anno[_data_anno['segment_id'] == _sids[_s]]
                _ti = int(_data_anno[_data_anno['segment_id'] == _sids[_s]].index.values)
                
                _data_anno = _data_anno.drop([_ti]).append(_tr).reset_index(drop=True)
                
            _data_anno['eda'] = _data_anno['eda'].apply(self._sequence_difference).apply(self._from_array_to_list)
                
            with open(f"{self._conf.embedding_data_path}/{_s_anno}_TimeSeries.pkl", "wb") as f:
                 pickle.dump(_data_anno, f, pickle.HIGHEST_PROTOCOL)
                 

if __name__ == "__main__" :
    ATP = TimeSeriesProcessor("./model/conf")
    ATP.make_data()