import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#%% Preparing the data

market = yf.download(tickers="^GSPC", start="1994-01-31", end = dt.date.today(), interval="1mo") # Download monthly data
market.columns = market.columns.get_level_values(0) # remove extra header
market = market.dropna(subset=['Close']).copy() # remove na values

#%% SMA and Signals

# --- Compute 10-month SMA and generate signals ---
market['SMA_10'] = market['Close'].rolling(window=10).mean() # Compute 10-month SMA
market['signal'] = (market['Close'] > market['SMA_10']).astype(int) # 1 when market above sma, 0 when below
market['position'] = market['signal'].shift(1).fillna(0).astype(int) # Shift signal to apply position for next month

#%% Returns

market['price_ret'] = market['Close'].pct_change().fillna(0) # Monthly price returns

try:
    tbill = yf.download("^IRX", start=market.index.min().strftime("%Y-%m-%d"),
                        end=(market.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval="1mo")['Close'] # Downloads 1-month U.S. Treasury Bill (T-Bill) rates (^IRX) from Yahoo Finance for the same date range as market
    tbill = tbill.reindex(market.index).ffill().bfill() # Aligns the T-Bill data with the market
    market['cash_ret'] = (tbill / 100) / 12 # Converts the annualized T-Bill yield (percent) to a monthly decimal return
except Exception:
    market['cash_ret'] = 0.0 # otherwise set to 0

market['strat_ret'] = market['position'] * market['price_ret'] + (1 - market['position']) * market['cash_ret'] # if we are long we get market return, if cash t-bill return

#%% Performance metrics

# Cumulative returns
market['cum_strat'] = (1 + market['strat_ret']).cumprod() # cummulative return for the strategy
market['cum_buyhold'] = (1 + market['price_ret']).cumprod() # cummulative return for buy and hold

def annualize_return(monthly_ret): # convert monthly returns to annualized
    # handle case with zero-length series
    if len(monthly_ret) == 0:
        return np.nan
    return (1 + monthly_ret).prod() ** (12 / len(monthly_ret)) - 1

def ann_vol(monthly_ret):
    return monthly_ret.std() * np.sqrt(12)

def max_drawdown(cum_returns):
    high = cum_returns.cummax()
    dd = (cum_returns - high) / high
    return dd.min()

# Compound return measures
cagr = annualize_return(market['strat_ret']) # average yearly growth rate of strategy
bh_cagr = annualize_return(market['price_ret']) # average yearly growth rate of buy and hold

# Total (cumulative) returns over full sample
total_ret_strat = market['cum_strat'].iloc[-1] - 1
total_ret_bh = market['cum_buyhold'].iloc[-1] - 1

# Final value of $100 invested at start
final_100_strat = 100 * market['cum_strat'].iloc[-1]
final_100_bh = 100 * market['cum_buyhold'].iloc[-1]

strat_vol = ann_vol(market['strat_ret']) # annualized volatility of strategy
bh_vol = ann_vol(market['price_ret']) # annualized volatility of buy and hold
sharpe = (cagr - market['cash_ret'].mean()*12) / strat_vol if strat_vol != 0 else np.nan # sharpe ration of strategy
mdd_strat = max_drawdown(market['cum_strat']) # maximum drawdown of strategy
mdd_bh = max_drawdown(market['cum_buyhold']) # market maximum drawdown
n_trades = market['signal'].diff().abs().sum() # number of trades using strategy 
last_position = market['position'].iloc[-1]

# Drawdown time series for plotting
market['dd_strat'] = market['cum_strat'] / market['cum_strat'].cummax() - 1
market['dd_bh'] = market['cum_buyhold'] / market['cum_buyhold'].cummax() - 1

#%% Visualization

# Metrics table
metrics = {
    'Metric': ['CAGR (strategy)', 'CAGR (buy & hold)', 
               'Total return (strategy)', 'Total return (buy & hold)',
               'Final $100 (strategy)', 'Final $100 (buy & hold)',
               'Annualized volatility (strategy)', 'Annualized volatility (buy & hold)',
               'Sharpe ratio', 'Max drawdown (strategy)', 'Max drawdown (buy & hold)',
               'Number of signals', 'Last position'],
    'Value': [f"{cagr:.2%}", f"{bh_cagr:.2%}",
              f"{total_ret_strat:.2%}", f"{total_ret_bh:.2%}",
              f"${final_100_strat:,.2f}", f"${final_100_bh:,.2f}",
              f"{strat_vol:.2%}", f"{bh_vol:.2%}",
              f"{sharpe:.2f}", f"{mdd_strat:.2%}", f"{mdd_bh:.2%}",
              int(n_trades), int(last_position)]
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Plot layout: 3 rows - cumulative linear (with buy/sell), cumulative log, drawdowns
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True)

# 1) Cumulative returns - linear scale with buy/sell markers
ax = axes[0]
ax.plot(market['cum_strat'], label='Strategy (cum)', linewidth=1.5)
ax.plot(market['cum_buyhold'], label='Buy & Hold (cum)', linewidth=1.5)
# Buy points (on cumulative strategy series)
buy_signals = market[(market['signal'] == 1) & (market['signal'].shift(1) == 0)]
ax.scatter(buy_signals.index, market.loc[buy_signals.index, 'cum_strat'], marker='^', s=80, label='Buy', zorder=5)
# Sell points
sell_signals = market[(market['signal'] == 0) & (market['signal'].shift(1) == 1)]
ax.scatter(sell_signals.index, market.loc[sell_signals.index, 'cum_strat'], marker='v', s=80, label='Sell', zorder=5)
ax.set_ylabel('Cumulative return (linear)')
ax.set_title('Cumulative returns (linear) - Strategy vs Buy & Hold')
ax.legend()
ax.grid(True)

# 3) Drawdowns over time
ax = axes[1]
ax.plot(market['dd_strat'], label='Strategy drawdown', linewidth=1.2)
ax.plot(market['dd_bh'], label='Buy & Hold drawdown', linewidth=1.2)
ax.set_ylabel('Drawdown')
ax.set_title('Drawdowns over time')
ax.legend()
ax.grid(True)

plt.xlabel('Date')
plt.tight_layout()
plt.show()

