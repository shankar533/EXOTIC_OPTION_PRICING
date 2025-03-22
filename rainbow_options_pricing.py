# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:00:49 2024

@author: shank
"""

import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np
from numpy.linalg import cholesky
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from whittaker_eilers import WhittakerSmoother
import seaborn as sns

df_treasury = pd.read_excel('US_TREASURY.xlsx')
#df_universal = pd.read_excel('US_UNIVERSAL.xlsx')
df_corporate = pd.read_excel('US_CORPORATE.xlsx')
df_risk_free_rates = pd.read_csv('US_3_month_treasury.csv')
df_risk_free_rates['Date'] = pd.to_datetime(df_risk_free_rates['Date'])
df_risk_free_rates['DTB3'] = df_risk_free_rates['DTB3'].replace('.', np.nan)
df_risk_free_rates['DTB3'] = df_risk_free_rates['DTB3'].astype(float)

df_treasury.set_index('Date', inplace = True)
#df_universal.set_index('Date', inplace = True)
df_corporate.set_index('Date', inplace = True)
df_risk_free_rates.set_index('Date', inplace = True)
df_treasury = df_treasury[~df_treasury.index.duplicated(keep='first')]
df_corporate = df_corporate[~df_corporate.index.duplicated(keep='first')]
df_risk_free_rates = df_risk_free_rates[~df_risk_free_rates.index.duplicated(keep='first')]

data_df = df_treasury.join([df_corporate,df_risk_free_rates], how='inner')
data_df=data_df.rename(columns={'Last_Price': 'treasury_price','Price':'corporate_price','DTB3': 'risk_free_rate'})
data_df['risk_free_rate'] = data_df['risk_free_rate'] / 100
data_df = data_df.dropna()
data_df = data_df[~data_df.index.duplicated(keep='first')]
data_df = data_df.iloc[::-1]

returns = pd.DataFrame()
returns['treasury'] = np.log(data_df['treasury_price'].pct_change() + 1)
#returns['universal'] = np.log(data_df['universal_price'].pct_change() + 1)
returns['corporate'] = np.log(data_df['corporate_price'].pct_change() + 1)
returns=returns.dropna()

daily_volatilities = returns.rolling(window=90).std() * np.sqrt(252)
daily_volatilities = daily_volatilities.dropna()

reduced_returns = returns.iloc[89:]

data_df = daily_volatilities.join(data_df, how='inner')
data_df = data_df.rename(columns={'treasury': 'treasury_vols','corporate': 'corporate_vols'})
data_df = data_df.join(returns, how='inner')
data_df = data_df.rename(columns={'treasury': 'treasury_ret','corporate': 'corporate_ret'})

correlation_matrix = data_df[['treasury_ret','corporate_ret']].corr()

def MC_rainbow(S0,sigma,r,price_paths,N,L,dt,T):
    # Simulate price paths
    np.random.seed(42)
    price_paths[0, :, :] = S0.reshape(-1, 1)
    price_paths = np.round(price_paths,10)
    for t in range(1, n_steps):
        Z = np.random.normal(size=(2, N))
        correlated_Z = L @ Z
        # Broadcasting adjustment: reshape sigma and apply the exponential correctly across all simulations
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma[:, np.newaxis] * np.sqrt(dt) * correlated_Z
        price_paths[t, :, :] = price_paths[t-1, :, :] * np.exp(drift[:, np.newaxis] + diffusion)
    
    
    strike_price = np.mean(S0) #Example strike price
    p=price_paths
    sim = price_paths[-1, :, :]
    bestof_payoffs = np.maximum(np.max(price_paths[-1, :, :], axis=0) - strike_price, 0)
    worstof_payoffs = np.maximum(np.min(price_paths[-1, :, :], axis=0) - strike_price, 0)
    # Discount the payoff to the present value
    discount_factor = np.exp(-r * T)
    best = np.median(bestof_payoffs)
    worst = np.median(worstof_payoffs)
    bestof_option_price = np.round(best * discount_factor,10)
    worstof_option_price = np.round( worst * discount_factor,10)
    #print(f"The estimated price of the Best-of Rainbow Option is: {option_price}")
    return bestof_option_price,worstof_option_price,strike_price,best,worst,sim,p 

T = 1  # Time to maturity in years
N = 1000  # Number of simulations
dt = 1/252  # Daily steps, assuming 252 trading days in a year
n_steps = int(T / dt)
corr_matrix = returns.corr().values # Correlation matrix and Cholesky decomposition
L = cholesky(corr_matrix)
price_paths = np.zeros((n_steps, 2, N))
max_call_payoff=[]
min_call_payoff=[]
max_call_price = []
min_call_price =[]
strike_mc = []
paths=[]
p=[]
for i in range(len(data_df)): 
    # Initial prices from the last available data using .iloc to avoid future warnings
    S0_treasury = data_df['treasury_price'][i]
    #S0_universal = data_df['universal_price'][i]
    S0_corporate = data_df['corporate_price'][i]
    S0 = np.array([S0_treasury,S0_corporate])
    
    # Annualized volatilities (latest available)
    sigma = daily_volatilities.iloc[i].values
    
    # Risk-free rate (latest available, annualized)
    r = data_df['risk_free_rate'].iloc[i]
    max_price,min_price,strike,best,worst,sim_paths,sim = MC_rainbow(S0,sigma,r,price_paths,N,L,dt,T)
    max_call_price.append(max_price)
    min_call_price.append(min_price)
    strike_mc.append(strike)
    max_call_payoff.append(best)
    min_call_payoff.append(worst)
    paths.append(sim_paths)
    p.append(sim)
data_df['MC_call_on_max_price'] = max_call_price
data_df['MC_call_on_min_price'] = min_call_price
data_df['Strike_price'] = strike_mc

def estimate_mean_reversion(returns):
    """
    Estimate the mean reversion rate (α) from historical returns using spectral analysis.
    
    Parameters:
    returns (pandas.Series): Historical returns of the underlying asset.
    
    Returns:
    float: Estimated mean reversion rate (α).
    """
    # Calculate the normalized fluctuation sequence
    Dn = np.cumsum(returns - returns.mean())
    Dn /= Dn.std()
    
    # Perform spectral analysis
    N = len(Dn)
    fft_Dn = fft(np.log(Dn))
    freqs = fftfreq(N, 1)
    
    # Find the peak frequency in the spectrum
    peak_freq = freqs[np.argmax(np.abs(fft_Dn))]
    
    # Estimate the mean reversion rate
    mean_reversion = 2 * np.pi * np.abs(peak_freq)
    
    return mean_reversion

def estimate_volatility_of_variance(returns, mean_reversion):
    """
    Estimate the volatility of variance (σv) from historical returns and the mean reversion rate.
    
    Parameters:
    returns (pandas.Series): Historical returns of the underlying asset.
    mean_reversion (float): Estimated mean reversion rate (α).
    
    Returns:
    float: Estimated volatility of variance (σv).
    """
    # Calculate the variance of the returns
    variance = returns.var()
    
    # Estimate the volatility of variance
    volatility_of_variance = np.sqrt(2 * mean_reversion * variance)
    
    return volatility_of_variance

def OU_rainbow_option_pricing2(S0,N,t,dt,corr,sigma,mean_reversion,T,K):
    S = np.zeros((N, len(S0), len(t)))
    S[:,:,0] = S0
    S=np.round(S,10)
    
    for i in range(1, len(t)):
        dW = np.random.multivariate_normal(np.zeros(len(S0)), corr * dt,N)
        sigma_t = sigma * np.exp(-mean_reversion * t[i])
        for j in range (N):
            S[j,:,i] = S[j,:,i-1] * np.exp((r - 0.5 * np.diag(sigma_t)) * dt + np.sqrt(sigma_t * dt) @ dW[j].T)
    
    # Calculate the best-of and worst-of rainbow option payoffs
    sim = S
    payoff_best_of = np.maximum(np.max(S[:,:,-1], axis=1) - K, 0)
    payoff_worst_of = np.maximum(np.min(S[:,:,-1], axis=1) - K, 0)
    best = np.median(payoff_best_of)
    worst = np.median(payoff_worst_of)
    # Calculate the option prices
    option_price_best_of = np.round(np.exp(-r * T) * best,10)
    option_price_worst_of = np.round(np.exp(-r * T) * worst,10)
    return option_price_best_of,option_price_worst_of,sim,best,worst

corr = correlation_matrix
N = 1000 # Number of simulation paths
dt = 1/252 # Time step
T = 1.0  # Time to maturity
t = np.arange(0, T, dt)
bestof=[]
worstof=[]
ou_call_max=[]
ou_call_min=[]
ou_sim_prices =[]

for i in range (len(data_df)):
    S0 = np.array([data_df['treasury_price'][i],data_df['corporate_price'][i]])
    K = S0.mean()
    r = data_df['risk_free_rate'][i]
    mean_reversion = estimate_mean_reversion(reduced_returns.iloc[i].values)
    long_term_variance = (daily_volatilities.iloc[i].values)**2
    volatility_of_variance = estimate_volatility_of_variance(reduced_returns.iloc[i].values, mean_reversion)
    sigma = np.diag(long_term_variance)
    maxcall,mincall,ou_path,best,worst = OU_rainbow_option_pricing2(S0,N,t,dt,corr,sigma,mean_reversion,T,K)
    ou_call_max.append(maxcall)
    ou_call_min.append(mincall)
    ou_sim_prices.append(ou_path)
    bestof.append(best)
    worstof.append(worst)
data_df['OU_call_on_max_price'] = ou_call_max
data_df['OU_call_on_min_price'] = ou_call_min

print("completed")

plt.figure(figsize=(14, 7))
plt.plot(data_df.index, np.array(data_df['MC_call_on_max_price']), label='MC_Call_on_MAX_prices ')
plt.plot(data_df.index, np.array(data_df['OU_call_on_max_price']), label='OU_Call_on_MAX_prices ',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Call on max price options of MC & OU ')
plt.title('MC vs OU')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data_df.index, (data_df['MC_call_on_max_price']-data_df['OU_call_on_max_price']), label='Diff_of_MAX_prices ')
plt.plot(data_df.index,(data_df['MC_call_on_min_price']-data_df['OU_call_on_min_price']) , label='Diff_of_MIN_prices ',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price Differences')
plt.title('Difference of Max vs Min')
plt.legend()
plt.show()

mc_whittaker_smoother = WhittakerSmoother(lmbda=1, order=2, data_length=len(data_df['MC_call_on_max_price']))
mc_smoothed_temp_anom = mc_whittaker_smoother.smooth(data_df['MC_call_on_max_price'])

ou_whittaker_smoother = WhittakerSmoother(lmbda=3, order=2, data_length=len(data_df['OU_call_on_max_price']))
ou_smoothed_temp_anom = ou_whittaker_smoother.smooth(data_df['OU_call_on_max_price'])

# Plot historical vs calibrated volatilities
plt.figure(figsize=(14, 7))
plt.plot(data_df.index, mc_smoothed_temp_anom, label='MC_Call_on_MAX_prices ')
plt.plot(data_df.index, ou_smoothed_temp_anom, label='OU_Call_on_MAX_prices ',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Call on max price options of MC & OU ')
plt.title('MC vs OU')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data_df.index, ((np.array(mc_smoothed_temp_anom)-np.array(ou_smoothed_temp_anom))), label='Diff_of_MAX_prices ')
plt.plot(data_df.index,data_df['MC_call_on_min_price']-data_df['MC_call_on_min_price'] , label='Diff_of_MIN_prices ',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price Differences')
plt.title('Difference of Max vs Min')
plt.legend()
plt.show()



plt.figure(figsize=(14,7))
plt.plot(data_df.index, data_df['treasury_price'], label='Treasury Prices', color='darkslategrey', linestyle='-')
plt.plot(data_df.index, data_df['corporate_price'], label='Corporate Prices', color='maroon', linestyle='--')
plt.xlabel('Date',fontsize=12, fontweight='bold',fontfamily='serif')
plt.ylabel('Price',fontsize=12, fontweight='bold',fontfamily='serif')
plt.title('I. Underlyings: Treasury vs. Corporate Prices Over Time',fontsize=13, fontweight='bold',fontfamily='serif')
plt.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.gca().set_facecolor('#f7f7f7')
sns.set(style="whitegrid")
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data_df.index, np.array(mc_smoothed_temp_anom), label='Correlated_GBM_Call_on_MAX_prices ', color ="maroon")
plt.plot(data_df.index, np.array(ou_smoothed_temp_anom), label='OU_Call_on_MAX_prices ', linestyle= '--',color ="darkslategrey")
plt.xlabel('Date',fontsize=12, fontweight='bold',fontfamily='serif')
plt.ylabel('Call on max price options of Correlated GBM & OU ',fontsize=12, fontweight='bold',fontfamily='serif')
plt.title('II. Rainbow Option Prices from Correlated GBM vs OU',fontsize=13, fontweight='bold',fontfamily='serif')

# Add legend with fancy box border
plt.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Customize background color
plt.gca().set_facecolor('#f7f7f7')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Show plot
plt.show()

# Plot historical vs calibrated volatilities
plt.figure(figsize=(14, 7))
plt.plot(data_df.index, np.array(mc_smoothed_temp_anom) - np.array(ou_smoothed_temp_anom), label='Difference betwen Correlated GBM & OU ', color = 'darkslategrey')
plt.xlabel('Date',fontsize=12, fontweight='bold',fontfamily='serif')
plt.ylabel('Difference in prices ',fontsize=12, fontweight='bold',fontfamily='serif')
plt.title('III. Rainbow Option Price differences: Correlated GBM vs OU',fontsize=13, fontweight='bold',fontfamily='serif')

# Add legend with fancy box border
plt.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Customize background color
plt.gca().set_facecolor('#f7f7f7')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Show plot
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have ETF prices in columns 'ETF1_price' and 'ETF2_price', and rainbow option prices in column 'Rainbow_option_price'
X = data_df['treasury_price']
Y = data_df['corporate_price']
Z = mc_smoothed_temp_anom

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_trisurf(X, Y, Z, cmap='coolwarm', edgecolor='none')

# Set labels and title
ax.set_xlabel('Treasury Price')
ax.set_ylabel('Corporate Price')
ax.set_zlabel('Rainbow Option Price')
ax.set_title('3D Surface Plot: Rainbow Option Price vs. ETF Prices')

#set view
#ax.view_init(elev=30, azim=30)

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot
plt.show()

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.

    Parameters:
    S (float): stock price
    K (float): strike price
    T (float): time to maturity in years
    r (float): risk-free rate
    sigma (float): volatility of the stock

    Returns:
    float: price of the call option
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the call price
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    
    return call_price

BS_tre_price=[]
BS_cor_price=[]
for i in range(len(data_df)):
    S0_tre = data_df['treasury_price'][i]
    S0_cor = data_df['corporate_price'][i]
    vol_tre = data_df['treasury_vols'][i]
    vol_cor = data_df['corporate_vols'][i]
    K = np.array([S0_tre,S0_cor])
    K = np.mean(K)
    r = data_df['risk_free_rate'][i]
    tre_price = black_scholes_call_price(S0_tre, K, 1, r, vol_tre)
    cor_price = black_scholes_call_price(S0_cor, K, 1, r, vol_cor)
    BS_cor_price.append(cor_price)
    BS_tre_price.append(tre_price)
    
data_df['Static_hedging_price'] = np.array(BS_cor_price) + np.array(BS_tre_price)

plt.figure(figsize=(14, 7))
plt.plot(data_df.index, data_df['MC_call_on_max_price']+data_df['MC_call_on_min_price'], label='Correlated_GBM_total_price ', color ="maroon")
plt.plot(data_df.index, data_df['OU_call_on_max_price']+data_df['OU_call_on_min_price'], label='OU_total_price ', color ="green",linestyle= 'dotted')
plt.plot(data_df.index, data_df['Static_hedging_price'], label='Static_hedging_price ', linestyle= '--',color ="darkslategrey")
plt.xlabel('Date',fontsize=12, fontweight='bold',fontfamily='serif')
plt.ylabel('Option prices of Correlated GBM, OU & static replication ',fontsize=12, fontweight='bold',fontfamily='serif')
plt.title('IV. Rainbow Option Prices of Correlated GBM, OU vs Vanilla Static replication Prices',fontsize=13, fontweight='bold',fontfamily='serif')

# Add legend with fancy box border
plt.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Customize background color
plt.gca().set_facecolor('#f7f7f7')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Show plot
plt.show()


plt.figure(figsize=(14, 7))
plt.plot(data_df.index, data_df['OU_call_on_max_price']+data_df['OU_call_on_min_price'], label='OU_total_price ', color ="maroon")
plt.plot(data_df.index, data_df['Static_hedging_price'], label='Static_hedging_price ', linestyle= '--',color ="darkslategrey")
plt.xlabel('Date',fontsize=12, fontweight='bold',fontfamily='serif')
plt.ylabel('Option prices of OU & static hedging ',fontsize=12, fontweight='bold',fontfamily='serif')
plt.title('IV. Rainbow Option Prices of OU vs Vanilla Static hedging Prices',fontsize=13, fontweight='bold',fontfamily='serif')

# Add legend with fancy box border
plt.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Customize background color
plt.gca().set_facecolor('#f7f7f7')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Show plot
plt.show()