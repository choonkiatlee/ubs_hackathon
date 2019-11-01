# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:58:58 2019

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_timeseries(csv_file, time_col_name = "DATE"):
    df = pd.read_csv(csv_file)
    df = set_time_index(df, time_col_name)
    df = df.sort_index()
    
    return df

def set_time_index(df, time_col_name = "DATE"):
    df = df.set_index(pd.to_datetime(df[time_col_name]))
    df = df.drop(columns = [time_col_name])
    
    return df

def merge_dfs(df1, df2, how = "left", method="ffill"):
    return df1.join(df2, how=how).fillna(method=method)

def normalise(series):
    return (series-series.mean())/series.std()