# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:42:25 2019

@author: Administrator
"""

import pandas as pd

import get_data

def get_relevant_CPI(CPI_filename="CPI.csv", frequency = "M", countries_to_collect=[]):

    cpi = pd.read_csv("CPI.csv")
    
    # Extract monthly CPI for relevant currencies
    monthly_cpi = cpi[ (cpi["FREQUENCY"] == frequency) and (cpi["LOCATION"] in countries_to_collect) ]
    
    return monthly_cpi