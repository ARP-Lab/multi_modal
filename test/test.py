# from model.DataPreprocess import delegate

# delegate()
import pandas as pd
import numpy as np

from preprocessing.pp import PreProcessing
from model.TimeSeriesProcessor import TimeSeriesProcessor
from model.AudioTextProcessor import AudioTextProcessor

ts = TimeSeriesProcessor()
# df = ts.make_data(19)
df = ts.make_data(20)
# at = AudioTextProcessor()
# df = at.make_data(19)

# print(df)

# pp = PreProcessing()
# pp.organize_data("./data")
# pp.test_org("./data")

# 2020-1018-1439-07-500
# 2020-1018-1439-07-750, 28.37