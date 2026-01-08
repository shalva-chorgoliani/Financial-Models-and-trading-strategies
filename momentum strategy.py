import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests

#%% 1. Prepare Data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"} 
html = requests.get(url, headers=headers).text 
sp500 = pd.read_html(html)[0] 
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).to_list() 

# Download Stock Data and Benchmark
start_date = "2009-01-01"
end_date = "2025-01-02"

data = yf.download(tickers, start=start_date, end=end_date, interval="1mo", auto_adjust=True)['Open']
benchmark = yf.download("^GSPC", start=start_date, end=end_date, interval="1mo", auto_adjust=True)['Open']

log_returns = np.log(data).diff()
benchmark_ret = np.log(benchmark).diff().cumsum() # Cumulative log returns for S&P 500

#%% 2. Backtest Engine
def run_strategy(lb, hp, returns_df):
    # Momentum: rolling sum of log returns
    # Shift(1) to avoid look-ahead bias
    mom = returns_df.rolling(lb).sum().shift(1)
    
    # Rebalance every 'hp' months
    rebalance_dates = returns_df.index[::hp]
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=returns_df.columns)
    
    for date in rebalance_dates:
        row = mom.loc[date]
        valid = row.dropna()
        if not valid.empty:
            k = max(1, int(len(valid) * 0.1)) # Top 10%
            winners = valid.rank(method='first') > (len(valid) - k)
            weights.loc[date, winners[winners].index] = 1.0 / k
            
    # Forward fill weights and align with returns
    full_weights = weights.reindex(returns_df.index).ffill()
    port_ret = (full_weights.shift(1) * returns_df).sum(axis=1)
    return port_ret.cumsum()

#%% 3. Run all 16 Combinations
lookbacks = [3, 6, 9, 12]
holdings = [3, 6, 9, 12]
all_results = pd.DataFrame()

for lb in lookbacks:
    for hp in holdings:
        name = f"L{lb}/H{hp}"
        all_results[name] = run_strategy(lb, hp, log_returns)

#%% 4. Visualization
plt.figure(figsize=(14, 8))

# Plot the 16 strategies using a colormap for better differentiation
colors = plt.cm.viridis(np.linspace(0, 1, 16))
for i, col in enumerate(all_results.columns):
    plt.plot(all_results[col], color=colors[i], alpha=0.6, linewidth=1, label=col)

# Plot Benchmark (S&P 500)
plt.plot(benchmark_ret, color='black', linewidth=3, linestyle='--', label='S&P 500 (Benchmark)')

plt.title("Comparison of 16 Momentum Strategies (Lookback/Holding) vs S&P 500", fontsize=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Cumulative Log Return", fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()