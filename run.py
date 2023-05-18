import argparse

import pandas as pd

from preprocessing.pp import PreProcessing

from model.AudioTextProcessor import AudioTextProcessor
from model.TimeSeriesProcessor import TimeSeriesProcessor

from model.ModelExec import ModelExec


def arg_parse():
    _p = argparse.ArgumentParser(description="multi_modal project")
    
    _p.add_argument('--make_data', type=bool, default=False)
    _p.add_argument('--all', type=bool, default=True)
    
    return _p.parse_args()


def data_preprocessing():
    _pp = PreProcessing()
    _pp.organize_data(root="./data", block_list=[])


def prepare_dataset():
    _atp = AudioTextProcessor()
    _tsp = TimeSeriesProcessor()
    
    _y19_at = _atp.make_data(year=19)
    _y20_at = _atp.make_data(year=20)
    _y19_ts = _tsp.make_data(year=19)
    _y20_ts = _tsp.make_data(year=20)
    
    return _y19_at, _y20_at, _y19_ts, _y20_ts
    

def run(
    emb_data_19: pd.DataFrame,
    emb_data_20: pd.DataFrame,
    anno_data_19: pd.DataFrame,
    anno_data_20: pd.DataFrame
) -> None:
    
    _me = ModelExec()
    _me.proc(emb_data_19, emb_data_20, anno_data_19, anno_data_20)


def proc():
    _args = arg_parse()
    
    if _args.make_data:
        data_preprocessing()
        _y19_at, _y20_at, _y19_ts, _y20_ts = prepare_dataset()
        
    elif _args.all:
        data_preprocessing()
        _y19_at, _y20_at, _y19_ts, _y20_ts = prepare_dataset()
        
        run(_y19_at, _y20_at, _y19_ts, _y20_ts)


if __name__ == "__main__":
    proc()