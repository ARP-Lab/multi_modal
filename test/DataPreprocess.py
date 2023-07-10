import warnings
warnings.filterwarnings('ignore')

import itertools
from itertools import accumulate
import re

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import csv

import pickle
from collections import Counter

from dfl.dfl import dfl_tools

r_path = "./data/org_KEMDy20"

ts_data_spices = ["EDA", "TEMP"]


root = '/home/arplab/project/paradeigma/multi_modal/'


def from_str_to_list(ts_lst):
    return ts_lst if isinstance(ts_lst, float) else [x for x in ts_lst]


def make_nan_empty_list(x):
    return [] if type(x) == type(np.NAN) else x


# def read_csv_info(
#     file_path: str=""
# ) -> pd.DataFrame:
    
#     csv_list = []
#     with open(file_path) as f:
#         r = csv.reader(f)

#         for l in r:
#             csv_list.append(l)

#     return pd.DataFrame(csv_list)


# def temp_read_data():
#     _t_path = "./data/KEMDy19/EDA/Session01/Original"
#     _t_df = read_csv_info(f"./data/KEMDy19/EDA/Session01/Original/Sess01F.csv")
#     _l = f"./data/KEMDy19/EDA/Session01/Original/Sess01F.csv".split("/")
#     _td = _l[len(_l) - 1]
    
#     _s_inv = "F" if _td.find("M") > 0 else "M"
#     _t_df = _t_df[_t_df[3].str.contains(_s_inv) == False]
#     _t_df.to_csv("wer.txt", sep="\t")
#     # _t_df.columns
    
#     # print(_t_df.to_string())
    

def make_data(path: str=""):
    assert path != ""
    
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
    
    _rdf = pd.DataFrame(columns=["Segment ID", "EDA", "TEMP", \
        "EDA length", "TEMP length", "Scaled EDA", "Scaled TEMP"] + \
            [f"{_}" for _ in _spices] + [f"{_}_vector" for _ in _spices])
    
    for _p in _sess_path:
        _sses = pd.read_csv(_p, index_col=0)
        # _sses.drop(columns=["ecg"], inplace=True)
        _sses.drop(columns=["ibi"], inplace=True)
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
            print(f"* anno : {_t_ap}")
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
    
    return _rdf
        

def make_data_y19():
    # ['{:02d}'.format(x) for x in range(1, 40)]
    session_lst = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    sex_lst = ['F', 'M']
    
    sid_dict = {}
    sid_lst = {}

    for _s in sex_lst:
        for session in session_lst:
            sid_lst[_s] = []
            df = pd.read_csv(root + f'/KEMDy19/annotation/Session{session}_{_s}_res.csv', skiprows=1)
            df.rename(columns = {'Unnamed: 9' : 'Segment ID'}, inplace = True)
            sid_dict[_s][session] = df['Segment ID']
            sid_lst[_s].extend(list(df['Segment ID']))

    sid_19_lst = sid_lst["M"].remove('Sess04_impro03_F031')
    
    
    anno_19_df = pd.DataFrame()
    missing_sid = {}

    for i, sid in enumerate(sid_19_lst):
        
        session = sid.split('_')[0][-2:]
        sex = sid.split('_')[2][0]
        # session 9에서 script 6번에서 여자가 첫번째로 말한 여자의 eda는 존재하지 않는다. 
        # 따라서, 아예 배제하기 보단 여자의 말을 들은 남자의 감정이, 즉 청자의 감정이 발화자와 같을 것이라고 가정한 뒤에 남자의 감정을 타겟으로 가져온다.
        if sid == 'Sess09_script06_F001':
            anno = pd.read_csv(root + f'/KEMDy19/annotation/Session{session}_M_res.csv', skiprows=1)
            
        else: 
            anno = pd.read_csv(root + f'/KEMDy19/annotation/Session{session}_{sex}_res.csv', skiprows=1)
            
        
        anno.rename(columns = {'Unnamed: 9' : 'Segment ID'}, inplace = True)
        anno = anno[anno.columns[9:]]
        
        # session 3에서 두 번째 즉흥적으로 한 여자의 첫 번째 발화는 남자쪽 eda에 두 번 기록이 되어있다. 그 두번 기록된 것 중 첫 번째 가 여자의 발화이다. 
        if sid in ['Sess03_impro02_F001', 'Sess09_impro01_F001']:
            id = sid.replace('F','M')
            row = anno[anno['Segment ID'] == sid].iloc[0]
            row['Segment ID'] = sid
        else: 
            row = anno[anno['Segment ID'] == sid]
        
        if len(row) == 0:
            missing_sid[i] = sid
        anno_19_df = anno_19_df.append(row, ignore_index=True)

    anno_19_df.reset_index(drop=True, inplace = True)
    
    temp = anno_19_df['Segment ID'].copy()
    temp[1362] = 'Sess03_impro02_F001'
    anno_19_df['Segment ID'] = temp
    
    anno_19_df.drop(1361, axis=0, inplace=True)
    anno_19_df.reset_index(drop=True, inplace=True)
    anno_19 = anno_19_df.drop(columns=list(anno_19_df.columns[4:]))
    
    _spices = ["emotion", "valence", "arousal"]
    _anno_data = {
        "emotion" : {
            "emotion_vector" : {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0},    
            "emotion_eval": ['Emotion.1','Emotion.2','Emotion.3','Emotion.4','Emotion.5','Emotion.6','Emotion.7','Emotion.8','Emotion.9','Emotion.10']
        },
        "valence" : {
            "valence_vector" : {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "valence_eval": ['Valence.1','Valence.2','Valence.3','Valence.4','Valence.5','Valence.6','Valence.7','Valence.8','Valence.9','Valence.10']
        },
        "arousal" : {
            "arousal_vector" : {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "arousal_eval": ['Arousal.1','Arousal.2','Arousal.3','Arousal.4','Arousal.5','Arousal.6','Arousal.7','Arousal.8','Arousal.9','Arousal.10']
        },
    }

    for _sp in _spices:
        eval_df = anno_19_df[_sp][f"{_sp}_eval"]
        
        for i in range(len(eval_df)):
            score_table = _anno_data[_sp][f"{_sp}_vector"]
            i_of_tsdata = Counter(eval_df.iloc[i])
            
            for val in i_of_tsdata.keys():
                score_table[val] += i_of_tsdata[val]
            anno_19[f"{_sp}_vector"].append(list(score_table.values()))
            
    tsdata_19_dict = {}

    for _tss in ts_data_spices:
        for sid in sid_19_lst:
            _v = []
            session = sid.split('_')[0][-2:]
            sex = sid.split('_')[2][0]            
                        
            with open(root + f'/KEMDy19/{_tss}/Session{session}/Original/Sess{session}{sex}.csv') as csvfile:
                eda_csv = csv.reader(csvfile)
                for row in eda_csv:
                    if len(row) > 3:
                        a, _, _, c = row
                        if c == sid:
                            _v.append(a)
                            
            tsdata_19_dict[f"{_tss}"][sid] = _v
            
    ts_19_df = pd.DataFrame(columns=['Segment ID', 'EDA', 'TEMP'])
    ts_19_df['Segment ID'] = tsdata_19_dict["EDA"].keys()
    ts_19_df['EDA'] = tsdata_19_dict["EDA"].values()
    ts_19_df['TEMP'] = tsdata_19_dict["TEMP"].values()
    
    # anno_19['EDA'] = ts_19_df['EDA']
    # anno_19['TEMP'] = ts_19_df['TEMP']
        
    # anno_19['EDA'] = anno_19['EDA'].apply(make_nan_empty_list)
    # anno_19['TEMP'] = anno_19['TEMP'].apply(make_nan_empty_list)
        
    anno_19['EDA'] = ts_19_df['EDA'].apply(make_nan_empty_list)
    anno_19['TEMP'] = ts_19_df['TEMP'].apply(make_nan_empty_list)
    
    anno_19['Emotion'] = anno_19['Emotion'].apply(lambda x: ';'.join(sorted(x.split(';'))))

    for _sp in ts_data_spices:
        anno_19[f'{_sp} length'] = anno_19[_sp].apply(lambda x:len(x))
        
        anno_19[_sp] = anno_19[_sp].apply(from_str_to_list)
        anno_19[f'Scaled {_sp}'] = 0
        anno_19[f'Scaled {_sp}'] = anno_19[f'Scaled {_sp}'].astype('object')
        
    session_script_sex = pd.Series([x[:-3] for x in sid_19_lst]).unique()
    length = len(anno_19)

    for _sp in ts_data_spices:
        _t_dict = {}
        
        for sss in session_script_sex:
            _t_value = []
            
            for segment in range(length):
                sid = anno_19['Segment ID'][segment]
                if sss in sid:
                    _t_value.extend(anno_19[_sp][segment])
                    
            if len(_t_value) != 0:  
                scaler = StandardScaler()  
                _t_value = scaler.fit_transform(np.array(_t_value).reshape(-1,1))   
                _t_dict[sss] = _t_value.reshape(1,-1).tolist()
                
            else: 
                _t_dict[sss] = [[]]
                
        for i, segment in enumerate(sid_19_lst):
            scaled_sss_eda = _t_dict[segment[:-3]][0]
            nums = int(anno_19[f'{_sp} length'][anno_19['Segment ID'] == segment])
            
            scaled_eda = scaled_sss_eda[:nums]
            _t_dict[segment[:-3]][0] = scaled_sss_eda[nums:]

            anno_19[f'Scaled {_sp}'][i] = scaled_eda
            
    missing_eda_index = set(anno_19[anno_19['EDA length'] == 0].index)
    missing_temp_index = set(anno_19[anno_19['TEMP length'] == 0].index)
    
    missing_ts_index = missing_eda_index | missing_temp_index
    anno_19_nonm = anno_19.loc[list(set(anno_19.index) - missing_ts_index)]
    anno_19_nonm.reset_index(drop=True, inplace=True)
    
    anno_19_nonm['Scaled EDA length'] = anno_19_nonm['Scaled EDA'].apply(lambda x: len(x))
    anno_19_nonm['Scaled TEMP length'] = anno_19_nonm['Scaled TEMP'].apply(lambda x: len(x))
    
    with open(root + '/model/data/paradeigma_KEMDY19_annotation_nonmissing.pkl', 'wb') as f:
        pickle.dump(anno_19_nonm, f, pickle.HIGHEST_PROTOCOL)
        

def delegate():
    make_data(r_path)
    # temp_read_data()
    
    
if __name__ == "__main__":
    make_data(r_path)