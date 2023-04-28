from typing import List
import os

import re
import numpy as np
import pandas as pd

import csv
import shutil

from utils.dfl import dfl_base, dfl_tools

def read_csv_info(
    file_path: str=""
) -> pd.DataFrame:
    
    csv_list = []
    with open(file_path) as f:
        r = csv.reader(f)

        for l in r:
            csv_list.append(l)

    return pd.DataFrame(csv_list)


def make_merged_data(
    target_dir_list: List[str],
    file_list: List[str],
    session_name: str,
    year: int=19
) -> pd.DataFrame:

    assert target_dir_list != []
    assert file_list != []
    assert session_name != ""
    assert not (year < 19 and year > 20)
    
    res: pd.DataFrame = None
    
    lable = ["eda", "temp", "ecg" if year == 19 else "ibi"]
    
    for _f in file_list:
        _fds = {}
        
        for tdp in target_dir_list:
            _td = tdp.split("/")
            
            # Directory path form name is as like "KEMDy19/EDA/Session01/Original/Sess01F.csv"
            # Directory path form name is as like "KEMDy20/EDA/Session01/Sess01_script01_User001F.csv"
            _t_df = read_csv_info(
                f"./{_td[0]}/{_td[1]}/{session_name}/"
                + "Original/" if year == 19 else "" + f"{_f}")
            
            if year == 19:
                _s_inv = "F" if _f.find("M") > 0 else "M"
                
                # processing for exceptional cases
                _fds[_td[1]] = _t_df[_t_df[3].str.contains(_s_inv) == False]
                            
                if _td[1] == "ECG":
                    _fds[_td[1]] = _fds[_td[1]].drop(columns=_fds[_td[1]].columns[0])
                else:
                    _fds[_td[1]] = _fds[_td[1]].drop(columns=_fds[_td[1]].columns[1])
                _fds[_td[1]] = _fds[_td[1]].T.reset_index(drop=True).T
                
            elif year == 20:
                _fds[_td[1]] = _t_df
                
                if not _fds[_td[1]].empty:
                    if _td[1] == "EDA" or _td[1] == "TEMP":
                        _fds[_td[1]] = _fds[_td[1]].drop(1) 
                    _fds[_td[1]] = _fds[_td[1]].drop(0)
                    
                    if _td[1] == "IBI":
                        _fds[_td[1]] = _fds[_td[1]].drop(columns=_fds[_td[1]].columns[0])
                        # reset a column index 
                        _fds[_td[1]] = _fds[_td[1]].T.reset_index(drop=True).T
                        
                    # drop the rows if has None
                    _fds[_td[1]] = _fds[_td[1]].dropna()
                    
                    _fds[_td[1]] = _fds[_td[1]].reset_index(drop=True)
                else:
                    if _td[1] == "IBI":
                        _fds[_td[1]] = pd.DataFrame({"IBI": [None], "timestamp": [None], "sid": [None]})
        
        # ** for check data
        # for x in target_dir_list:
        #     t = x.split("/")
        #     print(f"{t[1]} : ")
        #     print(file_datas[t[1]].to_string())
        
        # attatching column names
        for _l in lable:
            if year == 19:
                _fds[f"{_l.upper()}"] = _fds[f"{_l.upper()}"].set_axis([f"{_l}", "timestamp", "sid"], axis=1)
            elif year == 20:
                if len(_fds[f"{_l.upper()}"].columns) < 3:
                    _fds[f"{_l.upper()}"] = _fds[f"{_l.upper()}"].set_axis([f"{_l}", "timestamp"], axis=1)
                    _fds[f"{_l.upper()}"]["sid"] = np.nan
                else:
                    _fds[f"{_l.upper()}"] = _fds[f"{_l.upper()}"].set_axis([f"{_l}", "timestamp", "sid"], axis=1)

        # Merge "TEMP" table to "EDA"
        _opd = pd.merge(
            _fds[f"{lable[0].upper()}"], _fds[f"{lable[1].upper()}"],
            left_on='timestamp', right_on='timestamp', how='outer')
        del _opd["sid_x"]
        _opd = _opd.rename(columns={"sid_y": "sid"})

        # Merge "ECG" or "IBI" table to merged table("EDA" and "TEMP")
        _opd = pd.merge(
            _opd, _fds[f"{lable[2].upper()}"],
            left_on='timestamp', right_on='timestamp', how='outer')
        _opd["sid_x"] = _opd["sid_x"].fillna(_opd["sid_y"])
        del _opd["sid_y"]
        _opd = _opd.rename(columns={"sid_x": "sid"})

        # reorder following as "timestamp", "sid", "eda", "temp", "ecg" or "ibi"
        _opd = _opd[["timestamp", "sid"] + lable]
        # sorting values from "timestamp" column
        _opd = _opd.sort_values("timestamp")
        _opd = _opd.reset_index(drop=True)
        
        if res is None:
            res = _opd
        else:
            res = pd.concat([res, _opd], sort=True)
            # result = result.append(one_part_data_on_session, ignore_index=True)
            
    res = res[["timestamp", "sid"] + lable]
    res = res.sort_values("timestamp")
    res = res.reset_index(drop=True)
    
    # print(result.to_string())
    
    return res

def org_preprocessed_data(
    root: str="",
    year: int=19
) -> None:
    
    assert root != ""
    assert not (year < 19 and year > 20)
    
    os.system(f"./init y{year}")
    
    dir_name = f"KEMDy{year}"
    
    _y_num = re.findall(r"\d+", dir_name)
    _y_num = _y_num[0]
    origin_dp = f"{root}/{dir_name}"
    org_dp = f"{root}/org_{dir_name}"

    # make wav directory to root
    dfl_base.make_dir(".", org_dp)

    _wd = dfl_tools.find_dfl_path(f"./{dir_name}", ["wav"], mode="d", only_leaf="True")
    _sl = dfl_tools.find_dfl_path(f"{_wd[0]}", [""], mode="d", only_leaf="True", res_all=True)

    for l, r in _sl:
        dfl_base.make_dir(f"./{org_dp}", r)
        _ssl = dfl_tools.find_dfl_path(f"{l}", [""], mode="d", only_leaf="True")
        
        _mv_path = f"./{org_dp}/{r}"
        
        for _d in _ssl:
            _fl = dfl_tools.find_dfl_path(_d, [""], mode="f", only_leaf="True")
            for f in _fl:
                shutil.move(f, _mv_path)
                pass

    src_ap = os.path.join(os.getcwd(), f"{origin_dp}/annotation")
    dest_ap = os.path.join(os.getcwd(), f"{org_dp}/annotation")

    shutil.copytree(src_ap, dest_ap)

    target_dir_list = ["EDA", "ECG" if year == 19 else "IBI", "TEMP"]

    sd_list = []

    for _td in target_dir_list:
        _sdl = dfl_tools.find_dfl_path(f"./{origin_dp}/{_td}", ["Session"], mode="d", only_leaf=True, res_all=True)
        sd_list += [x[1] for x in _sdl]
        sd_list = list(set(sd_list))

    sd_list = sorted(sd_list)

    for _sn in sd_list:
        _tf = []
        
        for _td in target_dir_list:
            _sfl = dfl_tools.find_dfl_path(f"./{origin_dp}/{_td}/{_sn}", [""], mode="f", recur=True, only_leaf=True, res_all=True)
            _tf += [x[1] for x in _sfl]
            
        _tf = list(set(_tf))
        _tf = sorted(_tf)
        
        _smd = make_merged_data([f"{origin_dp}/{x}" for x in target_dir_list], _tf, _sn, _y_num)
        _smd.to_csv(f"{org_dp}/{_sn}/{_sn}.csv", sep=",", na_rep="NaN")
    
    
    # Doing after post-processing for exception on KEMDy19, 20 annotation datasets.
    _x = dfl_tools.find_dfl_path(
        f"{org_dp}/annotation", [".csv"],
        mode="f", recur=True,
        only_leaf=True, res_all=True)
    _nums = sorted(list(set(["".join(re.findall(r"\d+", _z)) for _, _z in _x])))

    for _n in _nums:
        _res_files = dfl_tools.find_dfl_path(
            f"{org_dp}/annotation", [f"{_n}", ".csv"],
            mode="f", recur=True,
            only_leaf=True, res_all=True)
        
        _rdf = None
        
        for _i, _j in enumerate(_res_files):
            _p, _fn = _j
            
            _an = pd.read_csv(_p, index_col=0, skiprows=1)
            _an.rename(columns={"Unnamed: 9" if int(_y_num) == 19 else " .1" : "Segment ID"},
                    inplace=True)
            _an = _an[_an.columns[8 if int(_y_num) == 19 else 3:]]
            _an.reset_index(drop=True, inplace=True)
            # print(_an)
                
            # 결국 수정할 것들은 다음과 같다.
            if int(_y_num) == 19:
                if _n == "03" and _fn.find("F"):
                    # Sess03_impro02_M001(여자쪽, 345) 수정
                    _t_row = _an.loc[_an["Segment ID"] == "Sess03_impro02_M001", "Segment ID"]
                    _t_row[345] = "Sess03_impro02_F001"
                    _an.loc[_an["Segment ID"] == "Sess03_impro02_M001", "Segment ID"] = _t_row
                elif _n == "09" and _fn.find("F"):
                    # Sess09_script06_M001(여자쪽, 253), Sess09_impro01_M001(여자쪽, 310) 수정
                    # 사실 이것은 잘못 들어온 Data 같아서 수정 대신 Drop 함
                    _an.drop(253, axis=0, inplace=True) # Sess09_script06_M001
                    _an.drop(310, axis=0, inplace=True) # Sess09_impro01_M001
                elif _n == "04":
                    # Sess04_impro03_M031(여자쪽, 467), Sess04_impro03_F031(남자쪽, 467) 삭제
                    _an.drop(467, axis=0, inplace=True) # Sess04_impro03_M031, Sess04_impro03_F031
            
            # 순서를 맞추기 위해 강제적으로 Segment ID 백업 뒤에 그대로 update하기 위해 만듦
            if _i == 0:
                _rdf = _an[_an.columns[:1]] # segment ID
                _empty_df = pd.DataFrame(columns=_an.columns[1:])
                _rdf = pd.concat([_rdf, _empty_df], axis=1)
                _rdf.set_index(keys=["Segment ID"], inplace=True, drop=False)
            
            if int(_y_num) == 19:
                _sx = ""
                
                if _fn.find("M") > -1: _sx = "F"
                elif _fn.find("F"): _sx = "M"
                    
                if _sx != "":
                    _an = _an[_an["Segment ID"].str.contains(_sx)]
            
            _an.set_index(keys=["Segment ID"], inplace=True, drop=False)
            _rdf.update(_an.set_index("Segment ID"))
            
        _rdf.reset_index(drop=True, inplace=True)
        _rdf.to_csv(f"{org_dp}/annotation/Session{_n}_res.csv", sep=",")
        
        for _p, _ in _res_files:
            os.remove(_p)

        
def organize_data(root: str="") -> None:
    assert root != ""
    
    years = [19, 20]
    
    for _y in years:
        org_preprocessed_data(root=root, year=_y)