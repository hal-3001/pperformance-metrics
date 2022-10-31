
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from scipy.optimize import minimize


def get_data(tickers,years):
    year=years*dt.timedelta(days=365)
    end=dt.date.today()
    start=end-year
    stock_prices=web.get_data_yahoo(tickers,start,end,interval="m")["Adj Close"]
    return stock_prices

def returns(data,plot=True,size=(10,5)):
    rets=data.pct_change()
    if plot:
        return rets.plot(figsize=size),rets.head(8)    
    else:
        return rets

    

def annualized_returns(returns,n_periods):
    compound_growth=(1+returns).prod()
    length=returns.shape[0]
    return compound_growth**(n_periods/length)-1

def annualized_volatility(returns,n_periods):
    return returns.std()*(n_periods**0.5)

def semi_deviation(returns,portfolio=False):
    if portfolio:
        less_than_zero=(returns<0)
        semi=returns[less_than_zero].fillna(0).cov()
    else:   
        less_than_zero=(returns<0)
        semi=returns[less_than_zero].std(ddof=0)
    return semi

def sharpe_ratio( returns, riskfree_rate,n_periods):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/n_periods)-1
    excess_ret = returns - rf_per_period
    ann_ex_ret = annualized_returns(excess_ret,n_periods)
    ann_vol = annualized_volatility(returns,n_periods)
    return ann_ex_ret/ann_vol

def portfolio_sharpe(portfolio_returns,portfolio_vol,r_free_rate):
    excess_ret=portfolio_returns-r_free_rate
    return excess_ret/portfolio_vol

def sortino_ratio(returns,riskfree_rate,n_periods):
    rf_per_period = (1+riskfree_rate)**(1/n_periods)-1
    excess_ret = returns - rf_per_period
    ann_ex_ret = annualized_returns(excess_ret,n_periods)
    ann_vol = annualized_volatility(semi_deviation(returns),n_periods)
    return ann_ex_ret/ann_vol


def portfolio_sortino(portfolio_returns,semi_d,r_free_rate):
    excess_ret=portfolio_returns-r_free_rate
    return excess_ret/semi_d

def maxdrawdown(returns):
    accumulated=1000*(1+returns).cumprod()
    maxwealth=accumulated.cummax()
    drawdown=(accumulated/maxwealth)-1
    return pd.DataFrame({"Wealth": accumulated, 
                         "Previous Peak": maxwealth, 
                         "Drawdown": drawdown})



def portfolio_returns(returns,weights):
    return (weights.T @ returns)



def portfolio_volatility(weights,covar,n_periods="m"):
    if n_periods=="m":
        return  (weights @ covar*(12)@ weights.T)**0.5
    if n_periods=="d":
         return  (weights @ covar*(256)@ weights.T)**0.5
    if n_periods=="y":
        return (weights @ covar @ weights.T)**0.5

def minimize_vol(target,ar,covar,n_periods):
    initial_guess=np.repeat(1/len(ar),len(ar))
    bounds=((0.0,1.0),)*len(ar)
    weights_equal_to_1={"type":"eq",
                       "fun":lambda weights:np.sum(weights)-1}
    is_target={"type":"eq",
                "args":(ar,),
                "fun":lambda ar,weights:target-portfolio_returns(weights,ar)}
    weights=minimize(portfolio_volatility,args=(covar,n_periods), x0=initial_guess
                    ,method="SLSQP",options={'disp': False},
                    constraints=(weights_equal_to_1,is_target),
                    bounds=bounds
                    )
    return weights.x

def weights_msr(ar,covar,n_periods,r_free_rate):
    initial_guess=[1/len(ar)]*len(ar)
    bounds=((0.0,1.0),)*len(ar)
    weights_equal_to_1={"type":"eq",
                       "fun":lambda weights:np.sum(weights)-1}
    
    def neg_sharpe(weights,ar,covar,n_periods,r_free_rate):
        rets=portfolio_returns(ar,weights)
        vol=portfolio_volatility(weights,covar,n_periods)
        return -((rets-r_free_rate)/vol)
        

    weights=minimize(neg_sharpe,initial_guess,args=(ar,covar,n_periods,r_free_rate),method="SLSQP"
    ,bounds=bounds,constraints=(weights_equal_to_1))
    
    return weights.x

def msr(ar,covar,n_periods,r_free_rate):
    weights=weights_msr(ar,covar,n_periods,r_free_rate)
    return weights,portfolio_returns(ar,weights),portfolio_volatility(weights,covar,n_periods)



def optimal_weights(ar,covar,n_periods,n_points):
    target_returns=np.linspace(ar.min(),ar.max(),n_points)
    weights=[minimize_vol(target_return,ar,covar,n_periods) for target_return in target_returns]
    return weights

def plot_optimal_portfolio(ar,covar,n_periods,n_points):
    weights=optimal_weights(ar,covar,n_periods,n_points)
    rets=[portfolio_returns(ar,weight) for weight in weights]
    risk=[portfolio_volatility(weight,covar,n_periods) for weight in weights]
    df=pd.DataFrame({"rets":rets,"risk":risk})
    return df.plot(x="risk",y="rets")

