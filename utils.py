# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:58:58 2019

@author: Administrator
"""

import pandas as pd

def read_timeseries(csv_file, time_col_name = "DATE"):
    df = pd.read_csv(csv_file)
    df = set_time_index(df, time_col_name)
    
    return df

def set_time_index(df, time_col_name = "DATE"):
    df = df.set_index(pd.to_datetime(df[time_col_name]))
    df = df.drop(columns = [time_col_name])
    
    return df