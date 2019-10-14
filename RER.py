# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:42:56 2019

@author: choon
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

import get_data
import utils

from sklearn.linear_model import LinearRegression

# Load FX data
if os.path.isfile(get_data.FX_spot_filename):    
    FX_spot = utils.read_timeseries(get_data.FX_spot_filename)

# Load FX Futures data
if os.path.isfile(get_data.FUT_filename):    
    FX_fut = utils.read_timeseries(get_data.FUT_filename)

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
    
    return (combined[ccy] * combined["USA_CPI"] / combined[ country + "_CPI"]).rename(ccy)

#def normalise(series, halflife = 3):
#    
#    ewm = series.ewm(halflife = halflife)   # 3 months half life
#    
#    return (series - ewm.mean()) / ewm.std()
    
def normalise(series):
    return (series-series.mean())/series.std()

#def backtest(signal, actual_price_series, plot_title="", debug = False):
#    
#    # Align signal to price series
#    aligned_signal = signal.align(actual_price_series, join='right')[0]
#        
#    # Calculate returns on the signal
#    returns = (actual_price_series.diff() * aligned_signal).rename("Cumulative Returns")
#    
#    plt.figure()
#    returns.cumsum().plot()
#    
#    if debug:
#        aligned_signal.plot()
#        actual_price_series.diff().plot()
#        
#    
#    # Calculate metrics?
#    aligned_price = actual_price_series.align(signal, join="left")[0]
#    
#    overall_return = returns.cumsum().iloc[-1]
#    
#    overall_std = (actual_price_series.diff() * aligned_signal).std()
#    
#    sharpe = overall_return / overall_std
#    
#    plt.title("{0} => sharpe: {1}".format(ccy, sharpe))
#    
#    plt.legend()
#    
#    plt.savefig("{0}.png".format(ccy))
    

def backtest(signal, actual_price_series, plot_title="", debug = False, plot_individual=True):
    
    # Align signal to price series
    aligned_signal = signal.align(actual_price_series, join='right')[0]
    
    # Purchased Weights
    portfolio_prices = aligned_signal * actual_price_series
        
    # Calculate returns on the signal
#    returns_1 = portfolio_prices.diff()
    
    returns = (actual_price_series.diff() * aligned_signal)   #.rename("Cumulative Returns")

        
    overall_return = returns.sum()
    overall_std = returns.std()
    sharpe = overall_return / overall_std
    
    print(sharpe)
    
    if plot_individual:
        for ccy in returns:
            
            plt.figure()
            
            
            if debug:
                normalise(returns[ccy].cumsum()).plot()
                normalise(aligned_signal[ccy]).rename("signal").plot()
                normalise(actual_price_series[ccy].diff()).rename("returns").plot()
            
            else:
                returns[ccy].cumsum().plot()
                returns[ccy].plot()
        
            plt.title("{0} => sharpe: {1}".format(ccy, sharpe[ccy]))
            plt.legend()
            plt.savefig("{0}.png".format(ccy))
    
         

# Due to a lack of data, we will use an expanding window with a minimum window size of 3months to do out of sample testing
ccy = "EUR"

investigated_ccys = [ "CAD","CHF","EUR","GBP","JPY","NOK","SEK"]


trading_signals = []

for ccy in investigated_ccys:
    
    print("Investingating {0}".format(ccy))
    
    min_window_size = 8 * 4   # 20 weeks = > approx 4 months
    horizon = 3
    
    RER = get_RER(ccy, monthly=True).dropna()
    
    starting_date = RER.index[0] + pd.Timedelta(weeks = min_window_size)
    ending_date_array = [ date for date in RER.index if date > starting_date]
    
    forecasted_spot = []
    forecasted_return = []
    dates = []
    
    for window_end_date in ending_date_array:
        
        windowed_RER = RER[:window_end_date - pd.Timedelta(weeks=1)]
    #    normalised = normalise(windowed_RER)
        
        diff = windowed_RER.diff(horizon).shift(-horizon).rename(ccy+"_diff")
        
        # Combine and drop nas
        combined = pd.concat([windowed_RER, diff], axis=1).dropna()
        
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(combined[ccy].values.reshape(-1,1), combined[ccy+"_diff"].values.reshape(-1,1))  # perform linear regression
        
        Y_pred = linear_regressor.predict(np.array([[RER[window_end_date]]]))  # make predictions
        
        forecasted_spot.append( combined[ccy].iloc[-1] - Y_pred[0][0] )
        forecasted_return.append(-Y_pred[0][0])
        
        dates.append(window_end_date)
        
    #    plt.scatter(windowed_RER, diff)
    #    plt.scatter([RER[window_end_date]], Y_pred, color='red')
    
    
#    forecasted_df.index = dates
    
#    forecasted_df = pd.DataFrame(forecasted_return, columns=["forecasted_return"], index=dates)
#    joined = RER.to_frame().join(forecasted_df, how='left')
#    #joined.plot()
#    
#    # Backtest
#    actual_return = joined[ccy].diff() * joined["forecasted_return"]
#    
#    plt.figure()
#    joined["forecasted_return"].plot()
#    joined[ccy].diff().plot()
#    actual_return.cumsum().plot()
#    plt.title(ccy)
#    
#    backtest(forecasted_df["forecasted_return"], RER, plot_title=ccy)
    
    forecasted_df = pd.DataFrame(forecasted_return, columns=[ccy], index=dates)
    trading_signals.append(forecasted_df)

combined_signal = pd.concat(trading_signals, sort=True, axis=1)
    
# Perform a mean variance optimisation
spots = pd.concat([FX_fut[ccy] for ccy in investigated_ccys], axis=1, sort=True).dropna()
spots = spots.align(combined_signal, join='right')[0]

cov = spots.ewm(halflife=6, min_periods = 6).std()   # Use shrinkage = 1 (i.e. ignore covariance)
MVO_optimised = combined_signal / cov
MVO_optimised.plot()

backtest(MVO_optimised, spots, plot_title=ccy, debug=False, plot_individual=True)




# Perform a mean variance optimisation
#RERs = pd.concat([get_RER(ccy, monthly=True) for ccy in investigated_ccys], axis=1, sort=True).dropna()
#
#cov = RERs.ewm(halflife=6, min_periods = 12).std()
#
#temp = np.zeros(combined_signal.shape)
#
#for i in range(combined_signal.shape[0]):
#    mean = combined_signal.iloc[i]
#    date = mean.name
#    current_cov = cov.loc[date]
#    
#    optimised = np.linalg.inv(current_cov.values) @ mean.values
#    
#    temp[i,:] = optimised
#    
#MVO_optimised = pd.DataFrame(temp[1:], index= combined_signal.index[1:], columns = combined_signal.columns)
#MVO_optimised.plot()
  
    
    
    


#for ccy in get_data.G10_ccys:
#    
#    if ccy == "USD":
#        continue
#    
#    RER = get_RER(ccy,monthly=True)["2010":].dropna()
#    
#    RER = (RER - RER.mean()) / RER.std()
#    
#    horizon = 3
#    
#    diff = RER.diff(horizon).shift(-horizon)
#    
#    if not RER.empty:
#        plt.figure()
#        plt.scatter(RER, diff)
#        plt.title(ccy)
#        plt.show()