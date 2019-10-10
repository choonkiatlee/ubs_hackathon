# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:55:18 2019

@author: Administrator
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

import get_data
import utils

import statsmodels.tsa.stattools as ts
from fbprophet import Prophet


# Load FX data
if os.path.isfile(get_data.FX_spot_filename):    
    FX_spot = utils.read_timeseries(get_data.FX_spot_filename)
    
relevant_ccys = FX_spot.columns
relevant_countries = [get_data.CCY_to_country_dict[ccy] for ccy in relevant_ccys]

# Collect relevant data from CPI
cpi = get_data.get_relevant_CPI(CPI_filename = get_data.CPI_filename, 
                          frequency = "M",
                          countries_to_collect=relevant_countries)

# Get arbritary country CPI
def get_country_CPI(country, subject = "TOT", cpi_table = cpi):
    country_cpi = cpi_table[ 
                    (cpi_table["LOCATION"] == country) & 
                    (cpi_table["SUBJECT"] == subject ) ]
    
    country_cpi = utils.set_time_index(country_cpi, "TIME")
    
    country_cpi = country_cpi.groupby(country_cpi.index).mean()
    country_cpi.columns = [ country + "_CPI"]
    
    return country_cpi

def merge_dfs(df1, df2, how = "left", method="ffill"):
    return df1.join(df2, how=how).fillna(method=method)


# Get US CPI
USA_cpi = get_country_CPI("USA")

def get_RER(ccy, FX_spot = FX_spot, cpi_table = cpi, USA_cpi = USA_cpi, monthly=False):

    country = get_data.CCY_to_country_dict[ccy]
    
    country_cpi = get_country_CPI(country)
    
    if monthly:
        combined = FX_spot[ccy].to_frame().join(USA_cpi, how="right")
        combined = combined.join(country_cpi)
    else:
        combined = merge_dfs(FX_spot[ccy].to_frame(), USA_cpi, method="bfill")
        combined = merge_dfs(combined, country_cpi, method="bfill")
    
    return combined[ccy] * combined["USA_CPI"] / combined[ country + "_CPI"]

for ccy in get_data.G10_ccys:
    
    if ccy == "USD":
        continue
    
    RER = get_RER(ccy,monthly=True)["2010":].dropna()
    
    RER = (RER - RER.mean()) / RER.std()
    
    horizon = 36
    
    diff = RER.diff(-horizon)
    
    if not RER.empty:
        plt.scatter(RER, diff)

#RER = get_RER("CAD", monthly=True)

#frame = RER.to_frame().reset_index()
#frame.columns = ["ds","y"]
#
#m = Prophet(changepoint_prior_scale=0.5, prior_scale=0.1)
#m.fit(frame)
#
#future = m.make_future_dataframe(periods=36, freq='M')
#
#forecast = m.predict(future)
#fig = m.plot_components(forecast)
#
#from fbprophet.diagnostics import cross_validation
#df_cv = cross_validation(m, initial='730 days', period='30 days', horizon = '365 days')
#df_cv.head()
#
#from fbprophet.plot import plot_cross_validation_metric
#fig = plot_cross_validation_metric(df_cv, metric='mape')




#results_dict = {}
#
#CAD_RER = get_RER("CAD", monthly=False)
#for ccy in get_data.G10_ccys:
#    if ccy != "USD":
#        RER = get_RER(ccy, monthly=True)["2009":]
#        
#        
##        RER = (RER - RER.mean()) / RER.std()
#        
#        RER = RER.dropna()
#
#        if not RER.empty:
#            
#            for i in range(0, len(RER) - 12*3):
#                test = RER.iloc[i:i+12*3]
#                                
#                # Test if series is mean reverting
#                adf_test = ts.adfuller( test, 1)
#                
#                if adf_test[0] < list(adf_test[4].values())[1]:
##                    print(ccy, adf_test[0], adf_test[4])
#                    
#                    results_dict[ccy] = results_dict.get(ccy,0) + 1
#                
#            print("Completed currency test for {0}".format(ccy))
                
            
        
        
#        if not RER.empty:
#                        
#            # Test if series is mean reverting
#            adf_test = ts.adfuller( RER, 1)
#            
#            if adf_test[0] < list(adf_test[4].values())[1]:
#                RER.plot()
#                print(ccy, adf_test[0], adf_test[4])
        
    


