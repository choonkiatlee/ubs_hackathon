# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:44:07 2019

@author: choon
"""

import pandas as pd
import matplotlib.pyplot as plt

from . import utils as utils

def backtest_additive(signal, actual_price_series, plot_title="", debug = False, plot_individual=True):
    
    # Align signal to price series
    aligned_signal = signal.align(actual_price_series, join='right')[0]
    
    # Purchased Weights
    portfolio_prices = aligned_signal * actual_price_series
        
    # Calculate returns on the signal
#    returns = portfolio_prices.diff()
    
    returns = (actual_price_series.diff() * aligned_signal)   #.rename("Cumulative Returns") # Should .shift(-1)!
    returns["overall"] = returns.sum(axis=1)

        
    overall_return = returns.sum()
    overall_std = returns.std()
    sharpe = overall_return / overall_std
    
    print(sharpe)
    
    if plot_individual:
        for ccy in returns:
            
            plt.figure()
                        
            if debug:
                utils.normalise(returns[ccy].cumsum()).plot()
                utils.normalise(aligned_signal[ccy]).rename("signal").plot()
                utils.normalise(actual_price_series[ccy].diff()).rename("returns").plot()
            
            else:
                returns[ccy].cumsum().plot()
#                returns[ccy].plot()
        
            plt.title("{0} => sharpe: {1}".format(ccy, sharpe[ccy]))
            plt.legend()
            plt.savefig("{0}.png".format(ccy))
            
            


def backtest(signal, actual_price_series, plot_title="", debug = False, plot_individual=True, save_fig=False):
    
    # First, we align the prices to the signal:
    aligned_prices = actual_price_series.align(signal, join="right")[0]
    raw_returns = (aligned_prices.shift(-1) / aligned_prices) - 1
    raw_returns["overall"] = ((raw_returns / raw_returns.shape[1])+1).product(axis=1) - 1
    #raw_returns = aligned_prices.diff().shift(-1)
    
    signal_returns = signal * raw_returns
    signal_returns["overall"] = (signal_returns+1).product(axis=1)-1
    
    cum_returns = (signal_returns+1).cumprod().dropna()
#    cum_returns.plot()
    
    overall_return = cum_returns.iloc[-1]
    overall_std = signal_returns.std()
    sharpe = (overall_return-1) / overall_std
    
    print(sharpe)
    
    if plot_individual:
        for ccy in signal_returns:
            
            plt.figure()
            
            if debug:
                utils.normalise(signal_returns[ccy].cumprod()).plot()
                utils.normalise(signal[ccy]).rename("signal").plot()
                utils.normalise(raw_returns).rename("returns").plot()
            
            else:
                cum_returns[ccy].plot(label="Strategy Returns")
                
                if ccy == "overall":
                    aligned_return = raw_returns.align(cum_returns,join="right")[0]
                    (aligned_return[ccy]+1).cumprod().plot(label="Buy and Hold Returns")
                
#                returns[ccy].plot()
        
            plt.title("{0} => sharpe: {1}".format(ccy, sharpe[ccy]))
            plt.legend()
            
            if save_fig:
                plt.savefig("{0}.png".format(ccy))



