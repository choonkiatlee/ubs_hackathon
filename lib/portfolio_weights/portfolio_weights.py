from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd

df = pd.read_csv("stock_prices.csv", index_col = "date")
print(df)

#Predictions goes here - mu would be a model of the expected returns, S would be a model of the covariance
mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()

#Weights optimisation goes here, after you have found some way to model mu and S
ef = EfficientFrontier(mu, S)

#If short positions available
#ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))

#Regularisation in order to encourage different non-zero weights
#ef = EfficientFrontier(mu, S, gamma=1)

#Other portfolio construction theories are:
#Hierarchical Risk Parity (HRP) - like a random forest with optimal weights at each branch
#Value-at-risk - minimises maximum expected shortfall within a given confidence

#Use different portfolio optimisers
weights = ef.max_sharpe()

'''
max_sharpe() - optimises for maximal Sharpe ratio (a.k.a the tangency portfolio)
min_volatility() - optimises for minimum volatility
custom_objective() - optimises for some custom objective function
efficient_risk() - maximises Sharpe for a given target risk
efficient_return() - minimises risk for a given target return
portfolio_performance() - calculates the expected return, volatility and Sharpe ratio for the optimised portfolio.
'''

cleaned_weights = ef.clean_weights()
print(cleaned_weights)

ef.portfolio_performance(verbose=True)