import pandas as pd
import os
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.optimize import minimize

from lib import get_data_second_round as get_data
from lib import utils as utils

asset_files = []
for file in os.listdir("Data"):
    if file.endswith(".xlsx"):
        name = os.path.join("Data", file)
        asset_files.append(name.replace("\\", "/"))

plt.figure
excel = asset_files[0]
df_main = pd.read_excel(excel, header = None, names = ["date", excel[5:-5]])
#df_main[excel[5:-5]] = (df_main[excel[5:-5]] - min(df_main[excel[5:-5]]))/max(df_main[excel[5:-5]])
df_main.set_index("date", drop = True, inplace = True)

for excels in asset_files[1:]:
    df = pd.read_excel(excels, header = None, names = ["date", excels[5:-5]])
#    df[excels[5:-5]] = (df[excels[5:-5]] - min(df[excels[5:-5]]))/max(df[excels[5:-5]])
    df.set_index("date", drop = True, inplace = True)
    df_main = df_main.join(df, how = "outer")

#print(df_main)
df_main = df_main.dropna()
#df_main.plot()
#plt.show()

#     plt.plot(df["date"], df["price"])
#     plt.xlabel("date")
#     plt.ylabel("price")
#     plt.title(excel[:-5])
# plt.show()

asset_allocation = {
        "BCOM":0.05,
        "BXIIBUS0":0.05,
        "dMIEF00000NUS":0.05,
        "dMIUS00000NUS":0.2,
        "dMIWOU0000NUS":0.15,
        "HFRXGL":0.1,
        "IBLUS0004":0.05,
        "JPMECORE":0.05,
        "JPMGABI":0.3,
        }

asset_allocation_df = pd.DataFrame(asset_allocation, index=[1])

# Every day, rebalance:
weights = df_main.asfreq('BM').copy()

for index in df_main.asfreq('BM').columns:
    weights[index] = asset_allocation[index] / df_main[index]
weights = weights.align(df_main,join='right')[0].fillna(method='ffill')
returns = (asset_allocation_df * df_main.pct_change().shift(-1) + 1)
returns["overall"] = returns.product(axis=1)
(returns.cumprod() * 100).plot()


## Collect a bunch of other tickers we might want to invest in
#rics = [
#        "VTI"
#        ]
#
#overlay_prices = get_data.get_historical_price(rics, 
#                                               "2007-01-01","2019-07-31", fields="BID, ASK", debug_printing=True)
#
#overlay_prices = overlay_prices.sort_index()
##overlay_prices = overlay_prices.where(overlay_prices < 110 ).where(overlay_prices > 70)
##overlay_prices[:300] = overlay_prices[:300].where(overlay_prices[:300] > 30)
#
#overlay_prices.plot()
###
#overlay_prices.to_csv("Data/vti_prices.csv")


files = [
#        "Data/lqd_price.csv",
#        "Data/sdy_price.csv",
#        "Data/dvy_prices.csv",
#        "Data/ief_prices.csv",
        "Data/tlt_prices.csv",
#        "Data/spy_prices.csv",
        "Data/vti_prices.csv",
        ]

dfs = [utils.read_timeseries(file).dropna() for file in files]
overlay_prices = pd.concat(dfs, axis=1).ffill()
#sdy_price[:1000] = sdy_price[:1000].where(sdy_price[:1000] < 70)
#sdy_price[:100] = sdy_price[:100].where(sdy_price[:100] > 38)
#
#sdy_price.to_csv("Data/sdy_price.csv")

overlay_prices = overlay_prices.sort_index()

overlay_returns = overlay_prices.pct_change().shift(-1)
overlay_returns = overlay_returns.sub(df_main["BXIIBUS0"].pct_change().shift(-1), axis=0)

portfolio = utils.read_timeseries("Data/portfolio.csv").dropna()
portfolio_returns = portfolio.shift(-1)/portfolio

overlay_returns["portfolio_returns"] = portfolio_returns - 1

#df_main = df_main.drop(columns="BXIIBUS0")


#overlay_returns = overlay_returns.align(df_main, join="right")[0]

 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    prev_x = pars[2]
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error 
    J = J + 0.0 * abs(x-prev_x)[:-1].sum() + 0.0 * abs(x)[:-1].sum()

    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

# At the end of each month, rebalance the portfolio using
monthly_df = overlay_returns.asfreq(freq='BM', method='ffill')

adjusted_weights = []
a = monthly_df.cov().values.shape[0]
w_rb = np.zeros(a)
w_rb[-1] = 1

for date in monthly_df.index[4:]:
    
    windowed_df = overlay_returns[date - pd.Timedelta(weeks=12):date]
    
    V = windowed_df.iloc[1:].cov().values * 100000
    w0 = np.ones(V.shape[0]) / V.shape[0]
    
    x_t = w0 # your risk budget percent of total portfolio risk (equal risk)
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res= minimize(risk_budget_objective, w0, args=[V,x_t, w_rb], method='SLSQP',constraints=cons, options={'disp': True})
    w_rb = res.x
    
    adjusted_weights.append(res.x)
    
adjusted_weight_df = pd.DataFrame(adjusted_weights, columns = overlay_returns.columns, index= monthly_df.index[4:])

aligned = adjusted_weight_df.align(overlay_returns,join='right')[0].fillna(method='ffill')

returns = (aligned * overlay_returns.shift(0) + 1)
returns["overall"] = returns.product(axis=1)
returns["original"] = portfolio_returns
(returns.cumprod() * 100).plot()

(returns.cumprod().iloc[-1])/100 / (returns.std() * np.sqrt(252))


adjusted_weight_df.plot()


#adjusted_weight_df = pd.DataFrame(adjusted_weights, columns=df_main.columns, index=monthly_df.index[4:])
#
#aligned = adjusted_weight_df.align(df_main,join='right')[0].fillna(method='ffill')
#
#returns = (aligned * df_main.pct_change().shift(0) + 1)
#returns["overall"] = returns.product(axis=1)
#(returns.cumprod() * 100).plot()




#from lib import get_data_second_round as get_data
#
#df_main = df_main.dropna()
#
#IR = get_data.get_historical_price(
#        ["US10YT=RR"],
#        "2007-01-01",
#        "2019-07-31", 
#        fields = "MID_YLD_1", 
#        debug_printing=False
#        )
#
#df_main["IR"] = IR
#
#correlations = df_main.corr()

    





