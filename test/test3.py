import pickle

import pandas as pd
import numpy as np

import itertools
from itertools import accumulate

import ast

import csv


def save_csv(year: str="19"):
    path = "./data/csv"

    with open(f"{path}/paradeigma_KEMDY{year}_annotation_nonmissing.pkl", "rb") as f:
        _a = pickle.load(f)
        _pd = pd.DataFrame(_a)
    _pd.to_csv(f"{path}/origin_ts_data_{year}.csv")


def get_empty_value(year: str="19"):
    path = "./data/csv"

    _pda = pd.read_csv(f"{path}/origin_ts_data_{year}.csv", index_col=0)
    _pdb = pd.read_csv(f"{path}/ts_data_{year}.csv", index_col=0)
    
    _la = _pda["Segment ID"].to_list()
    _lb = _pdb["Segment ID"].to_list()
    
    _lx = []
    for l in _la:
        if l not in _lb:
            _lx.append(l)
    
    print(_lx)
    
    
def make_sorted_tables(year: str="19"):
    path = "./data/csv"

    _pda = pd.read_csv(f"{path}/origin_ts_data_{year}.csv", index_col=0)
    _pdb = pd.read_csv(f"{path}/ts_data_{year}.csv", index_col=0)
    
    _pda.sort_values(by="Segment ID", inplace=True)
    _pdb.sort_values(by="Segment ID", inplace=True)
    
    print(_pda)
    _pda.drop(columns=["Emotion", "Valence", "Arousal", "emotion_vector", "valence_vector", "arousal_vector", "EDA length", "TEMP length", "Scaled EDA length" , "Scaled TEMP length"], inplace=True)
    _pdb.drop(columns=["EDA length", "TEMP length"], inplace=True)
    
    _pda.reset_index(drop=True, inplace=True)
    _pdb.reset_index(drop=True, inplace=True)
    
    _la = _pda["TEMP"].to_list()
    _lb = _pdb["TEMP"].to_list()
    
    print(len(_la))
    print(len(_lb))
    
    _odd_list = []
    cnt = 0
    
    for x, y in zip(_la, _lb):
        if len(x) != len(y):
            _odd_list.append(cnt)
        else:
            for _dx, _dy in zip(x, y):
                if _dx != _dy:
                    _odd_list.append(cnt)
                    break
        
        cnt += 1
        
    print(_odd_list)
    _pda.to_csv(f"{path}/ts_data_{year}_o.csv")
    _pdb.to_csv(f"{path}/ts_data_{year}_c.csv")


def make_temp_tables(year: str="19"):
    path = "./data/csv"

    _pda = pd.read_csv(f"{path}/origin_ts_data_{year}.csv", index_col=0)
    _pdb = pd.read_csv(f"{path}/ts_data_{year}.csv", index_col=0)
    
    _pda.sort_values(by="Segment ID", inplace=True)
    _pdb.sort_values(by="Segment ID", inplace=True)
    
    print(_pda)
    _pda.drop(columns=["Emotion", "Valence", "Arousal", "emotion_vector", "valence_vector", "arousal_vector", "EDA length", "TEMP length", "Scaled EDA length" , "Scaled TEMP length", "EDA", "Scaled EDA", "Scaled TEMP"], inplace=True)
    _pdb.drop(columns=["EDA length", "TEMP length", "EDA", "Scaled EDA", "Scaled TEMP"], inplace=True)
    
    _pda.reset_index(drop=True, inplace=True)
    _pdb.reset_index(drop=True, inplace=True)
    
    _la = _pda["TEMP"].to_list()
    _lb = _pdb["TEMP"].to_list()
    
    print(len(_la))
    print(len(_lb))
    
    _odd_list = []
    cnt = 0
    
    for x, y in zip(_la, _lb):
        if len(x) != len(y):
            _odd_list.append(cnt)
        else:
            for _dx, _dy in zip(x, y):
                if _dx != _dy:
                    _odd_list.append(cnt)
                    break
        
        cnt += 1
        
    print(_odd_list)
    _pda.to_csv(f"{path}/ts_data_{year}_t_o.csv")
    _pdb.to_csv(f"{path}/ts_data_{year}_t_c.csv")
    
    
def make_temp_tables(year: str="19"):
    path = "./data/csv"

    _pda = pd.read_csv(f"{path}/origin_ts_data_{year}.csv", index_col=0)
    _pdb = pd.read_csv(f"{path}/ts_data_{year}.csv", index_col=0)
    
    _pda.sort_values(by="Segment ID", inplace=True)
    _pdb.sort_values(by="Segment ID", inplace=True)
    
    print(_pda)
    _pda.drop(columns=["Emotion", "Valence", "Arousal", "emotion_vector", "valence_vector", "arousal_vector", "EDA length", "TEMP length", "Scaled EDA length" , "Scaled TEMP length", "EDA", "Scaled EDA", "Scaled TEMP"], inplace=True)
    _pdb.drop(columns=["EDA length", "TEMP length", "EDA", "Scaled EDA", "Scaled TEMP"], inplace=True)
    
    _pda.reset_index(drop=True, inplace=True)
    _pdb.reset_index(drop=True, inplace=True)
    
    _la = _pda["TEMP"].to_list()
    _lb = _pdb["TEMP"].to_list()
    
    print(len(_la))
    print(len(_lb))
    
    _odd_list = []
    cnt = 0
    
    for x, y in zip(_la, _lb):
        if len(x) != len(y):
            _odd_list.append(cnt)
        else:
            for _dx, _dy in zip(x, y):
                if _dx != _dy:
                    _odd_list.append(cnt)
                    break
        
        cnt += 1
        
    print(_odd_list)
    _pda.to_csv(f"{path}/ts_data_{year}_st_o.csv")
    _pdb.to_csv(f"{path}/ts_data_{year}_st_c.csv")


def check_diff(year: str="19"):
    path = "./data/csv"

    _pda = pd.read_csv(f"{path}/ts_data_{year}_st_o.csv", index_col=0)
    _pdb = pd.read_csv(f"{path}/ts_data_{year}_st_c.csv", index_col=0)
    
    _la = _pda["TEMP"].to_list()
    _lb = _pdb["TEMP"].to_list()
    
    # print(len(_la))
    # print(len(_lb))
    
    _odd_list = []
    cnt = 0
    
    for x, y in zip(_la, _lb):
        if len(x) != len(y):
            _odd_list.append(cnt)
        else:
            for _dx, _dy in zip(x, y):
                if _dx != _dy:
                    _odd_list.append(cnt)
                    break
        
        cnt += 1


def _scaling(vec: list):
    from sklearn.preprocessing import StandardScaler
            
    _sc = StandardScaler()
    
    # _t = _sc.fit_transform(np.array(vec).reshape(-1, 1))
    # _x = _t.reshape(1, -1).tolist()
    # print(_x)
    # return _x
    
    return list(itertools.chain(*_sc.fit_transform(np.array(vec).reshape(-1, 1)).reshape(1, -1).tolist()))


def make_scaled_data(year: str="19"):
    path = "./data/csv"

    _pda = pd.read_csv(f"{path}/ts_data_{year}_st_o.csv", index_col=0)
    _pdb = pd.read_csv(f"{path}/ts_data_{year}_st_c.csv", index_col=0)
    
    _la = _pda["Segment ID"].to_list()
    _lb = _pdb["Segment ID"].to_list()
    
    # _lx = _pda["TEMP"].to_list()
    # print(_lx)
    
    _ca = sorted(list(set([_n[:6].replace("Sess", "") for _n in _la])))
    _cb = sorted(list(set([_n[:6].replace("Sess", "") for _n in _lb])))
    
    # print(_ca)
    
    _rda = pd.DataFrame(columns=["Segment ID", "TEMP", "Scaled TEMP"])
    _rdb = pd.DataFrame(columns=["Segment ID", "TEMP", "Scaled TEMP"])
    
    _re_name = []
    _re_temp = []
    _re_stemp = []
    
    for _n in _ca:
        # print(_n)
        # T
        # _re_name_p = []
        # _re_temp_p = []
        # _re_stemp_p = []
        # _rdt = pd.DataFrame(columns=["Segment ID", "TEMP", "Scaled TEMP"])
        
        for _sx in ["F", "M"]:
            _x = _pda[_pda["Segment ID"].str.contains(f"Sess{_n}")]
            _x = _x[_x["Segment ID"].str.contains(f"{_sx}")]
            
            if _x.empty:
                continue
            
            _ss_list = _x["Segment ID"].to_list()
            _ss_list = sorted(list(set([_[:-4] for _ in _ss_list])))
            
            for _ss in _ss_list:
                _xs = _x[_x["Segment ID"].str.contains(f"{_ss}")]
                # _xs.sort_values(by="Segment ID", inplace=True)
                # _xs.reset_index(drop=True, inplace=True)
            
                # print(_xs)
                _temp_list = _xs["TEMP"].to_list()
                _temp_list = [ast.literal_eval(_) for _ in _temp_list]
                
                _temp_cnt = [len(_) for _ in _temp_list]
                
                _temp_list = sum(_temp_list, [])
                
                # print(f"{_ss}: {sum(_temp_cnt)}")
                # print(_temp_list)
                # print(_temp_cnt)
                from sklearn.preprocessing import StandardScaler
                
                _sc = StandardScaler()
                _tv = list(itertools.chain(
                    *_sc.fit_transform(np.array(_temp_list).reshape(-1, 1)).reshape(1, -1).tolist()))
                # _tv = _scaling(_temp_list)
                
                _re_stemp += [_tv[x - y: x] for x, y in zip(accumulate(_temp_cnt), _temp_cnt)]
                _re_temp += _xs["TEMP"].to_list()
                _re_name += _xs["Segment ID"].to_list()
                del _sc
                del _tv
                
                # break
            
            # break
        
        # _rdt["Segment ID"] = _re_name_p
        # _rdt["TEMP"] = _re_temp_p
        # _rdt["Scaled TEMP"] = _re_stemp_p
        # _rdt.sort_values(by="Segment ID", inplace=True)
        # _rdt.reset_index(drop=True, inplace=True)
        
        # _re_name += _re_name_p
        # _re_temp += _re_temp_p
        # _re_stemp += _re_stemp_p
        # print(_rdt)
        # break
        
    _rda["Segment ID"] = _re_name
    _rda["TEMP"] = _re_temp
    _rda["Scaled TEMP"] = _re_stemp
    _rda.sort_values(by="Segment ID", inplace=True)
    _rda.reset_index(drop=True, inplace=True)
    
    # print(_rda)
    _rda.to_csv(f"{path}/ts_data_{year}_ev_st_o.csv")
        
        

# get_empty_value("19")
# get_empty_value("20")

# make_sorted_tables("20")
# make_temp_tables("20")

# check_diff("20")

make_scaled_data("20")
