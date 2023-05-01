import warnings
warnings.filterwarnings('ignore')

import itertools
from itertools import accumulate
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from collections import Counter

from zconf.zconf import zconf
from dfl.dfl import dfl_base, dfl_tools

import pickle
import csv


class TimeSeriesProcessor(zconf):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_id=zconf_id)
        
    
    def make_data(
        self,
        year: int=None
    ) -> pd.DataFrame:
        
        assert year is not None, "\"year\" param must input."
        assert not (year < 19 and year > 20)
        
        path = self.glob_conf["org_data_path"][year]
        assert path is not ("" or None)
        
         # open a file to Session*.csv
        _res_path = dfl_tools.find_dfl_path(
            f"{path}", ["Session", r"[0-9]+", "csv"],
            mode="f", cond="a",
            recur=True, only_leaf=True, res_all=False)
        
        _sess_path = []
        _anno_path = []
        
        for _s in _res_path:
            _p, _ = _s
            if _p.find("anno") == -1:
                _sess_path.append(_p)
            else:
                _anno_path.append(_p)
            
        _spices = ["emotion", "valence", "arousal"]
        
        _opt_regression = [f"{_}" for _ in _spices] + [f"{_}_vector" for _ in _spices] \
            if self.local_conf["regression"] else []
        _rdf = pd.DataFrame(columns=["Segment ID", "EDA", "TEMP", \
            "EDA length", "TEMP length", "Scaled EDA", "Scaled TEMP"] + _opt_regression)
        
        for _p in _sess_path:
            _sses = pd.read_csv(_p, index_col=0)
            _sses.drop(columns=["ecg" if year == 19 else "ibi"], inplace=True)
            _sses = _sses.dropna()
            _sses.reset_index(drop=True, inplace=True)
            
            _sid_list = [name for name, _ in _sses.groupby("sid")]
            _sid_uq = []
            
            # Organize EDA and TEMP to zipped each vectors.
            for _sid in _sid_list:
                _tk = _sid.split("_")
                _str = f"{_tk[0]}_{_tk[1]}"
                
                if _str not in _sid_uq:
                    _sid_uq.append(_str)
                
                _tx = _sses[_sses["sid"].eq(_sid)]
                
                _tdf = pd.DataFrame()
                _tdf["Segment ID"] = [f"{_sid}"]
                
                for _sp in ["eda", "temp"]:
                    _tdf[_sp.upper()] = [_tx[_sp].to_list()]
                    _tdf[f"{_sp.upper()} length"] = _tdf[_sp.upper()].apply(lambda x:len(x))
            
                _rdf = pd.concat([_rdf, _tdf])
            
            _rdf = _rdf.set_index(keys=["Segment ID"], inplace=False, drop=False)
            
            # calculate to Scaled EDA, TEMP
            for _suq in _sid_uq:
                for _sx in ["M", "F"]:
                    _rx = _rdf[_rdf["Segment ID"].str.contains(_suq)]
                    _rx = _rx[_rx["Segment ID"].str.contains(_sx)]
                    
                    if _rx.empty:
                        continue

                    for _sp in ["EDA", "TEMP"]:
                        _l = sum(_rx[_sp].to_list(), [])
                        
                        # using Standardscale for make scaled EDA and TEMP
                        _sc = StandardScaler()
                        _tv = list(itertools.chain(*_sc.fit_transform(np.array(_l).reshape(-1, 1)).reshape(1,-1).tolist()))
                        _len_list = _rx[f"{_sp} length"].to_list()
                        
                        _rx[f"Scaled {_sp}"] = [_tv[x - y: x] for x, y in zip(accumulate(_len_list), _len_list)]                    
                        # print(f"{_suq}, {_sx}, {_sp}")

                    _rdf.update(_rx.set_index("Segment ID"))
            
            if self.local_conf["regression"]:
                # for Regression(Optional)
                # Extract of number for find path from annotation path list.
                _x = _p.split("/")
                _x = [_ for _ in _x[len(_x) - 1].split(".")][0]
                _ns = "".join(re.findall(r"\d+", _x))
                
                _aps = []
                for _t_ap in _anno_path:
                    _lp = _t_ap.split("/")
                    _aps += list(filter(None, [_t_ap if all(_i in _lp[len(_lp) - 1] for _i in [f"{_ns}", ".csv"]) else []]))

                for _t_ap in _aps:
                    _anno = pd.read_csv(_t_ap, index_col=0)
                    _anno.rename(columns={'Unnamed: 9' : 'Segment ID'}, inplace=True)
                    # _anno = _anno[_anno.columns[8:]]
                    _anno.reset_index(drop=True, inplace=True)
                    
                    from enum import IntEnum
                    class EmotionEnum(IntEnum):
                        angry = 1; disgust = 2; fear = 3; happy = 4;
                        neutral = 5; sad = 6; surprise = 7
                    
                    _anno.set_index(keys=["Segment ID"], inplace=True, drop=False)
                    _anno.rename(columns={f"{_.title()}":f"{_}" for _ in _spices}, inplace=True)
                    _t_anno = _anno
                    _anno = _anno[_anno.columns[:1]]
                    _empty_df = pd.DataFrame(columns=[f"{x}" for x in _spices]+[f"{x}_vector" for x in _spices])
                    
                    _anno = pd.concat([_anno, _empty_df], axis=1)
                    
                    for _sx in ["M", "F"]:
                        _rx = _t_anno[_t_anno["Segment ID"].str.contains(_sx)]
                        _rx.reset_index(drop=True, inplace=True)
                    
                        for _sp in _spices:    
                            _ev_target_val = _rx[[f"{_sp.title()}.{_}" for _ in range(1, 10 + 1)]]
                            _sp_vec_list = []
                            
                            for _i in range(len(_ev_target_val)):
                                _score_table = {f"{EmotionEnum(_).name}" if _sp == "emotion" else _:0 for _ in range(1, (7 if _sp == "emotion" else 5) + 1)}
                                _data = Counter(_ev_target_val.iloc[_i])
                                
                                for _e in _data.keys():
                                    _score_table[_e] += _data[_e]
                                
                                _sp_vec_list.append(list(_score_table.values()))
                                
                            _rx[f"{_sp}_vector"] = pd.Series(_sp_vec_list)
                        
                        # concat column to column.
                        _rx = pd.concat([_rx[_rx.columns[:4]], _rx[_rx.columns[34:]]], axis=1)
                        _anno.update(_rx.set_index("Segment ID"))
                        
                    _rdf.update(_anno.set_index("Segment ID"))
        
        _rdf.dropna()
        _rdf.reset_index(drop=True, inplace=True)
        
        if self.local_conf["save_pickle"]:
            with open(self.glob_conf["data_path"] + "/pkl" + f"/ts_data_{year}.pkl", "wb") as f:
                pickle.dump(_rdf, f, pickle.HIGHEST_PROTOCOL)
                
        if self.local_conf["save_csv"]:
            _rdf.to_csv(self.glob_conf["data_path"] + "/csv" + f"/ts_data_{year}.csv", sep=",")
        
        return _rdf