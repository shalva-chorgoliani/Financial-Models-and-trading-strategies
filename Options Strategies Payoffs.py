import numpy as np
import matplotlib.pyplot as plt

#%% Options Strategies 

def protective_put(S0, K, P, S_min=0.5, S_max=1.5, points=200):
    """
    ------------------------------------------------
    Buy a stock andd ATM/OTM put with the K<S0
    This strategy gives you ownership of the stock and different greeks profile
    This is a hedging strategy: the put option hedges the risk of the stock price falling
    ------------------------------------------------
    S0      initial stock price
    K       strike price
    P       option premium
    S_min   lower bound multiplier for price range
    S_max   upper bound multiplier
    points  number of grid points in the plot
    """

    S = np.linspace(S_min * S0, S_max * S0, points)

    payoff = (S - S0) + np.maximum(K - S, 0) - P # payoff at expiry
    breakeven = S0 + P # breakeven
    maxloss = S0 - K + P

    print("Break even:", breakeven,
          "Max loss:", maxloss)

    # plot
    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Protective Put Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(breakeven, linestyle="--", label=f"Break even {breakeven:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Protective Put at Expiry")
    plt.show()

    return S, payoff, breakeven

# Example:
protective_put(100, 100, 3)

def long_straddle(K, C, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: long ATM call otion and long ATM put option, with the strike price K
    ---------------------------------------------
    K       strike of call and put
    P       premium paid for put
    C       premium paid for call
    """

    S = np.linspace(S_min * K, S_max * K, points)

    payoff = np.maximum(S - K, 0) + np.maximum(K - S, 0) - (C + P)

    up_be = K + C + P
    down_be = K - (C + P)
    maxloss = C + P

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long Straddle Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Long Straddle at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Examole:
long_straddle(100, C=4, P=3.5)

def long_strangle(K1, K2, C, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: long OTM call otion with strike price K1 
    and long OTM put option, with the strike price K2.
    This strategy is less costly to establish than a long straddle, however BEPs are higher 
    ---------------------------------------------
    K1      strike of call
    K2      strike of put
    P       premium paid for put
    C       premium paid for call
    """

    S = np.linspace(S_min * K2, S_max * K1, points)

    payoff = np.maximum(S - K1, 0) + np.maximum(K2 - S, 0) - (C + P)

    up_be = K1 + C + P
    down_be = K2 - (C + P)
    maxloss = C + P

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long Strangle Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Long Strangle at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

# Example:
long_strangle(105, 95, 2, 2)

def long_guts(K1, K2, C, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: long ITM call otion with strike price K1 
    and long ITM put option, with the strike price K2.
    This strategy is less costly to establish than a long straddle. The outlook is neutrall
    Assuming that P + C > K2 - K1
    ---------------------------------------------
    K1      strike of call
    K2      strike of put
    P       premium paid for put
    C       premium paid for call
    """

    S = np.linspace(S_min * K2, S_max * K1, points)

    payoff = np.maximum(S - K1, 0) + np.maximum(K2 - S, 0) - (C + P)

    up_be = K1 + C + P
    down_be = K2 - (C + P)
    maxloss = (C + P) - (K2-K1)

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long guts Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Long guts at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

# Example:
long_guts(95, 105, 6, 6)

def short_straddle(K, C, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Sideway strategy: short ATM call otion and short ATM put option.
    The traders oulook is neutral
    ---------------------------------------------
    K       strike
    P       premium paid for put
    C       premium paid for call
    """

    S = np.linspace(S_min * K, S_max * K, points)

    payoff = -np.maximum(S - K, 0) - np.maximum(K - S, 0) + (C + P)

    up_be = K + C + P
    down_be = K - (C + P)
    maxprofit = C + P

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Profit:", maxprofit)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Short Straddle Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Short Straddle at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

# Example:
short_straddle(100, 1.5, 1.5)

def short_strangle(K1, K2, C, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Sideway strategy: short OTM call otion with strike K1 and short OTM put option with strike K2.
    This strategy is less risky, but the initial credit is lower
    The traders oulook is neutral
    ---------------------------------------------
    K1      strike of call
    K2      strike of put
    P       premium paid for put
    C       premium paid for call
    """

    S = np.linspace(S_min * K2, S_max * K1, points)

    payoff = -np.maximum(S - K1, 0) - np.maximum(K2 - S, 0) + (C + P)

    up_be = K1 + C + P
    down_be = K2 - (C + P)
    maxprofit = C + P

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Profit:", maxprofit)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Short strangle Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Short strangle at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Example:
short_strangle(105, 95, 1, 1)

def short_guts(K1, K2, C, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: short ITM call otion with strike price K1 
    and short ITM put option, with the strike price K2.
    The initial credit is higher
    ---------------------------------------------
    K1      strike of call
    K2      strike of put
    P       premium paid for put
    C       premium paid for call
    """

    S = np.linspace(S_min * K2, S_max * K1, points)

    payoff = -np.maximum(S - K1, 0) - np.maximum(K2 - S, 0) + (C + P)

    up_be = K1 + C + P
    down_be = K2 - (C + P)
    maxprofit = (C + P) - (K2 - K1)

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxprofit)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Short guts Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Short guts at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Example:
short_guts(105, 95, 5, 5)

def long_call_synthetic_straddle(K, S0, C, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: short stock and buy 2 ATM calls
    The outlook is neutral
    ---------------------------------------------
    K       strike of call
    S0      stock price
    C       premium paid for call
    """

    S = np.linspace(S_min * K, S_max * K, points)

    payoff = S0 - S + 2 * np.maximum(S - K, 0) - 2*C

    up_be = 2 * K - S0 + C
    down_be = S0 - C
    maxloss = 2*C - (S0 - K)

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long call synthetic straddle Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Long call synthetic straddle at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Example:
long_call_synthetic_straddle(100, 100, 1.5)

def long_put_synthetic_straddle(K, S0, P, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: buy stock and buy 2 ATM puts
    The outlook is neutral
    ---------------------------------------------
    K       strike of put
    S0      stock price
    P       premium paid for put
    """

    S = np.linspace(S_min * K, S_max * K, points)

    payoff = S - S0 + 2 * np.maximum(K - S, 0) - 2*P

    up_be = S0 + P
    down_be = 2*K - S0 - P
    maxloss = 2*P - (K - S0)

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long put synthetic straddle Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Long put synthetic straddle at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Example:
long_put_synthetic_straddle(100, 98, 2.5)

def strap(K, P, C, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: buy 2 ATM calls and buy ATM put with strike K
    The outlook is bullish
    ---------------------------------------------
    K       strike of put
    C       premium paid for call
    P       premium paid for put
    """

    S = np.linspace(S_min * K, S_max * K, points)

    payoff = 2 * np.maximum(S - K, 0) + np.maximum(K - S, 0) - (2*C + P)

    up_be = K + (2*C + P)/2
    down_be = K - (2*C + P)
    maxloss = (2*C + P)

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Strap Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Strap at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Example:
strap(100, 1.5, 1.5)

def strip(K, P, C, S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Volatility strategy: buy ATM calls and buy 2 ATM put with strike K
    The outlook is bullish
    ---------------------------------------------
    K       strike of put
    C       premium paid for call
    P       premium paid for put
    """

    S = np.linspace(S_min * K, S_max * K, points)

    payoff = np.maximum(S - K, 0) + 2 * np.maximum(K - S, 0) - (C + 2*P)

    up_be = K + (C + 2*P)
    down_be = K - (C + 2*P)/2
    maxloss = (C + 2*P)

    print("Upper break even:", up_be)
    print("Lower break even:", down_be)
    print("Max Loss:", maxloss)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Strip Payoff")
    plt.axhline(0, linewidth=1)
    plt.axvline(up_be, linestyle="--", label=f"Upper {up_be:.2f}")
    plt.axvline(down_be, linestyle="--", label=f"Lower {down_be:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.title("Strip at Expiry")
    plt.show()

    return S, payoff, up_be, down_be

#Example:
strip(100, 1.5, 1.5)

def call_ratio_backspread(K1, K2, C, P, NS=1, NL=2,
                          S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Short Ns ATM call options with strike K1, long Nl OTM calls with K2
    with Nl>Ns
    Traders outlook is strongly bullish
    ---------------------------------------------

    K1   strike of short calls
    K2   strike of long calls
    C    premium received per short call
    P    premium paid per long call
    NS   number of short calls
    NL   number of long calls, must be > NS

    Net premium = NL*P − NS*C
    Payoff = NL*(S − K2)+ − NS*(S − K1)+ − net premium
    """

    if NL <= NS:
        raise ValueError("NL must be greater than NS")

    net_prem = NL * P - NS * C

    S = np.linspace(S_min * K1, S_max * K2, points)
    payoff = NL * np.maximum(S - K2, 0) - NS * np.maximum(S - K1, 0) - net_prem

    S_down = K1 - net_prem / NS
    S_up = (NL * K2 - NS * K1 + net_prem) / (NL - NS)
    L_max = NS * (K2 - K1) + net_prem

    print("S_down:", S_down)
    print("S_up:", S_up)
    print("Max loss:", L_max)
    print("Max profit: unlimited")

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Call Ratio Backspread")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_down, linestyle="--", label=f"Down {S_down:.2f}")
    plt.axvline(S_up, linestyle="--", label=f"Up {S_up:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Call Ratio Backspread at Expiry")
    plt.legend()
    plt.grid(True)
    plt.show()

    return S, payoff, S_down, S_up, L_max

# Example
call_ratio_backspread(K1=100, K2=110, C=1.2, P=0.3)

def put_ratio_backspread(K1, K2, C, P, NS=1, NL=2,
                         S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Short Ns ATM put options with strike K1,
    Long Nl OTM puts with strike K2 (K2 < K1)
    with Nl > Ns
    Trader's outlook is strongly bearish
    ---------------------------------------------

    K1   strike of short puts
    K2   strike of long puts (OTM)
    C    premium received per short put
    P    premium paid per long put
    NS   number of short puts
    NL   number of long puts, must be > NS

    Net premium = NS*C − NL*P
    Payoff = NL*(K2 − S)+ − NS*(K1 − S)+ − net premium
    """

    if NL <= NS:
        raise ValueError("NL must be greater than NS")

    net_prem = NS * C - NL * P

    S = np.linspace(S_min * K2, S_max * K1, points)

    payoff = NL * np.maximum(K2 - S, 0) - NS * np.maximum(K1 - S, 0) - net_prem

    S_up = K1 + net_prem / NS
    S_down = (NL * K2 - NS * K1 - net_prem) / (NL - NS)
    L_max = NS * (K1 - K2) - net_prem
    P_max = NL * K2 - NS * K1 - net_prem

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max loss:", L_max)
    print("Max profit:", P_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Put Ratio Backspread")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--", label=f"Up {S_up:.2f}")
    plt.axvline(S_down, linestyle="--", label=f"Down {S_down:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Put Ratio Backspread at Expiry")
    plt.legend()
    plt.grid(True)
    plt.show()

    return S, payoff, S_up, S_down, L_max, P_max

# Example
put_ratio_backspread(K1=100, K2=95, C=1.2, P=0.5)

def ratio_call_spread(K1, K2, C, P, NS=2, NL=1,
                      S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Ratio Call Spread
    Short NS near-ATM calls at strike K1
    Long  NL ITM calls at strike K2 (K2 < K1)
    with NL < NS
    Trader outlook: neutral to bearish
    ---------------------------------------------

    K1   strike of short calls
    K2   strike of long calls (ITM)
    C    premium received per short call
    P    premium paid per long call
    NS   number of short calls
    NL   number of long calls, must be < NS

    Net premium = NS*C − NL*P
    Payoff = NL*(S − K2)+ − NS*(S − K1)+ − net premium
    """

    if NL >= NS:
        raise ValueError("NL must be strictly less than NS")

    net_prem = NS * C - NL * P

    S = np.linspace(S_min * K2, S_max * K1, points)

    payoff = NL * np.maximum(S - K2, 0) - NS * np.maximum(S - K1, 0) - net_prem

    S_down = K2 + net_prem / NL
    S_up = (NS * K1 - NL * K2 - net_prem) / (NS - NL)
    P_max = NL * (K1 - K2) - net_prem
    L_max = float("inf")

    print("S_down:", S_down)
    print("S_up:", S_up)
    print("Max profit:", P_max)
    print("Max loss: unlimited")

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Ratio Call Spread")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_down, linestyle="--", label=f"Down {S_down:.2f}")
    plt.axvline(S_up, linestyle="--", label=f"Up {S_up:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Ratio Call Spread at Expiry")
    plt.legend()
    plt.grid(True)
    plt.show()

    return S, payoff, S_down, S_up, P_max

# Example
ratio_call_spread(K1=100, K2=90, C=1.5, P=2.2, NS=2, NL=1)

def ratio_put_spread(K1, K2, C, P, NS=2, NL=1,
                     S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Ratio Put Spread
    Short NS near ATM puts at strike K1
    Long  NL ITM puts at strike K2 (K2 > K1)
    with NL < NS
    Trader outlook: neutral to bullish
    ---------------------------------------------

    K1   strike of short puts
    K2   strike of long puts (ITM)
    C    premium received per short put
    P    premium paid per long put
    NS   number of short puts
    NL   number of long puts, must be < NS

    Net premium = NS*C − NL*P
    Payoff = NL*(K2 − S)+ − NS*(K1 − S)+ − net premium
    """

    if NL >= NS:
        raise ValueError("NL must be strictly less than NS")

    net_prem = NS * C - NL * P

    S = np.linspace(S_min * K1, S_max * K2, points)

    payoff = NL * np.maximum(K2 - S, 0) - NS * np.maximum(K1 - S, 0) - net_prem

    S_up = K2 - net_prem / NL
    S_down = (NS * K1 - NL * K2 + net_prem) / (NS - NL)
    P_max = NL * (K2 - K1) - net_prem
    L_max = NS * K1 - NL * K2 + net_prem

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Ratio Put Spread")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--", label=f"Up {S_up:.2f}")
    plt.axvline(S_down, linestyle="--", label=f"Down {S_down:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Ratio Put Spread at Expiry")
    plt.legend()
    plt.grid(True)
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max

# Example
ratio_put_spread(K1=100, K2=110, C=1.5, P=2.0, NS=2, NL=1)

def long_call_butterfly(K1, K2, K3, C1, C2, C3,
                        S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Long Call Butterfly
    Long  1 OTM call at K1
    Short 2 ATM calls at K2
    Long  1 ITM call at K3
    Strikes equidistant: K2-K3 = K1-K2 = kappa
    Trader outlook: neutral
    ---------------------------------------------

    K1  upper strike (OTM)
    K2  middle strike (ATM)
    K3  lower strike (ITM)

    C1  premium paid for K1 call
    C2  premium received per K2 call
    C3  premium paid for K3 call

    Net debit D = C1 + C3 - 2*C2
    Payoff = (S-K1)+ + (S-K3)+ - 2*(S-K2)+ - D
    """

    if not np.isclose(K1 - K2, K2 - K3):
        raise ValueError("Strikes must be equidistant")

    kappa = K1 - K2
    D = C1 + C3 - 2 * C2

    S = np.linspace(S_min * K3, S_max * K1, points)

    payoff = (
        np.maximum(S - K1, 0)
        + np.maximum(S - K3, 0)
        - 2 * np.maximum(S - K2, 0)
        - D
    )

    S_down = K3 + D
    S_up = K1 - D
    P_max = kappa - D
    L_max = D

    print("S_down:", S_down)
    print("S_up:", S_up)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long Call Butterfly")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_down, linestyle="--", label=f"Down {S_down:.2f}")
    plt.axvline(S_up, linestyle="--", label=f"Up {S_up:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Long Call Butterfly at Expiry")
    plt.legend()
    plt.grid(True)
    plt.show()

    return S, payoff, S_down, S_up, P_max, L_max


# Example
long_call_butterfly(
    K1=110, K2=100, K3=90,
    C1=1.2, C2=3.0, C3=6.5
)

def long_put_butterfly(K1, K2, K3, P1, P2, P3,
                       S_min=0.5, S_max=1.5, points=200):
    """
    ---------------------------------------------
    Long Put Butterfly
    Long  1 OTM put at K1
    Short 2 ATM puts at K2
    Long  1 ITM put at K3
    Strikes equidistant: K3-K2 = K2-K1 = kappa
    Trader outlook: neutral
    It is similar to long call buterfly but uses puts
    ---------------------------------------------

    K1  lower strike (OTM)
    K2  middle strike (ATM)
    K3  upper strike (ITM)

    P1  premium paid for K1 put
    P2  premium received per K2 put
    P3  premium paid for K3 put

    Net debit D = P1 + P3 - 2*P2
    Payoff = (K1-S)+ + (K3-S)+ - 2*(K2-S)+ - D
    """

    if not np.isclose(K3 - K2, K2 - K1):
        raise ValueError("Strikes must be equidistant")

    kappa = K2 - K1
    D = P1 + P3 - 2 * P2

    S = np.linspace(S_min * K1, S_max * K3, points)

    payoff = (
        np.maximum(K1 - S, 0)
        + np.maximum(K3 - S, 0)
        - 2 * np.maximum(K2 - S, 0)
        - D
    )

    S_up = K3 - D
    S_down = K1 + D
    P_max = kappa - D
    L_max = D

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long Put Butterfly")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--", label=f"Up {S_up:.2f}")
    plt.axvline(S_down, linestyle="--", label=f"Down {S_down:.2f}")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Long Put Butterfly at Expiry")
    plt.legend()
    plt.grid(True)
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max


# Example
long_put_butterfly(
    K1=90, K2=100, K3=110,
    P1=1.1, P2=3.0, P3=6.8
)

def short_call_butterfly(K1, K2, K3, C1, C2, C3,
                         S_min=0.5, S_max=1.5, points=200):
    """
    Short Call Butterfly
    Short 1 ITM call at K1
    Long  2 ATM calls at K2
    Short 1 OTM call at K3
    Net credit strategy
    """

    if not np.isclose(K3 - K2, K2 - K1):
        raise ValueError("Strikes must be equidistant")

    kappa = K2 - K1
    C = 2 * C2 - C1 - C3   # net credit

    S = np.linspace(S_min * K1, S_max * K3, points)

    payoff = (
        2 * np.maximum(S - K2, 0)
        - np.maximum(S - K1, 0)
        - np.maximum(S - K3, 0)
        + C
    )

    S_up = K3 - C
    S_down = K1 + C
    P_max = C
    L_max = kappa - C

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Short Call Butterfly")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--")
    plt.axvline(S_down, linestyle="--")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Short Call Butterfly at Expiry")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max

# Example
short_call_butterfly(K1=90, K2=100, K3=110, C1=6.5, C2=3.0, C3=1.2)


def short_put_butterfly(K1, K2, K3, P1, P2, P3,
                        S_min=0.5, S_max=1.5, points=200):
    """
    Short Put Butterfly
    Short 1 ITM put at K1
    Long  2 ATM puts at K2
    Short 1 OTM put at K3
    Net credit strategy
    """

    if not np.isclose(K1 - K2, K2 - K3):
        raise ValueError("Strikes must be equidistant")

    kappa = K1 - K2
    C = 2 * P2 - P1 - P3   # net credit

    S = np.linspace(S_min * K3, S_max * K1, points)

    payoff = (
        2 * np.maximum(K2 - S, 0)
        - np.maximum(K1 - S, 0)
        - np.maximum(K3 - S, 0)
        + C
    )

    S_down = K3 + C
    S_up = K1 - C
    P_max = C
    L_max = kappa - C

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Short Put Butterfly")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--")
    plt.axvline(S_down, linestyle="--")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Short Put Butterfly at Expiry")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max


# Examples
short_put_butterfly(K1=110, K2=100, K3=90, P1=6.8, P2=3.0, P3=1.1)

def long_iron_butterfly(K1, K2, K3, P1, P2, C2, C3,
                        S_min=0.5, S_max=1.5, points=200):
    """
    Long Iron Butterfly (net credit)
    Long  OTM put at K1
    Short ATM put at K2
    Short ATM call at K2
    Long  OTM call at K3
    """

    if not np.isclose(K2 - K1, K3 - K2):
        raise ValueError("Strikes must be equidistant")

    kappa = K2 - K1
    C = P2 + C2 - P1 - C3   # net credit

    S = np.linspace(S_min * K1, S_max * K3, points)

    payoff = (
        np.maximum(K1 - S, 0)
        - np.maximum(K2 - S, 0)
        - np.maximum(S - K2, 0)
        + np.maximum(S - K3, 0)
        + C
    )

    S_up = K2 + C
    S_down = K2 - C
    P_max = C
    L_max = kappa - C

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Long Iron Butterfly")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--")
    plt.axvline(S_down, linestyle="--")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Long Iron Butterfly at Expiry")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max

# Example
long_iron_butterfly(
    K1=90, K2=100, K3=110,
    P1=1.1, P2=3.0, C2=3.2, C3=1.0
)


def short_iron_butterfly(K1, K2, K3, P1, P2, C2, C3,
                         S_min=0.5, S_max=1.5, points=200):
    """
    Short Iron Butterfly (net debit)
    Short OTM put at K1
    Long  ATM put at K2
    Long  ATM call at K2
    Short OTM call at K3
    """

    if not np.isclose(K2 - K1, K3 - K2):
        raise ValueError("Strikes must be equidistant")

    kappa = K2 - K1
    D = P1 + C3 - P2 - C2   # net debit

    S = np.linspace(S_min * K1, S_max * K3, points)

    payoff = (
        np.maximum(K2 - S, 0)
        + np.maximum(S - K2, 0)
        - np.maximum(K1 - S, 0)
        - np.maximum(S - K3, 0)
        - D
    )

    S_up = K2 + D
    S_down = K2 - D
    P_max = kappa - D
    L_max = D

    print("S_up:", S_up)
    print("S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Short Iron Butterfly")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--")
    plt.axvline(S_down, linestyle="--")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Short Iron Butterfly at Expiry")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max


# Examples
short_iron_butterfly(
    K1=90, K2=100, K3=110,
    P1=1.1, P2=3.0, C2=3.2, C3=1.0
)

def long_call_condor(K1, K2, K3, K4, C1, C2, C3, C4, points=200):
    """
    ---------------------------------------------
    Long Call Condor
    Long  ITM call at K1
    Short ITM call at K2
    Short OTM call at K3
    Long  OTM call at K4
    Strikes equidistant: K4-K3 = K3-K2 = K2-K1 = kappa
    Net debit strategy
    Trader outlook: neutral
    Capital gain strategy
    ---------------------------------------------

    Net debit D = C1 - C2 - C3 + C4
    Payoff = (S-K1)+ - (S-K2)+ - (S-K3)+ + (S-K4)+ - D
    """

    kappa = K2 - K1
    D = C1 - C2 - C3 + C4
    S = np.linspace(0.5*K1, 1.5*K4, points)

    payoff = (
        np.maximum(S-K1,0)
        - np.maximum(S-K2,0)
        - np.maximum(S-K3,0)
        + np.maximum(S-K4,0)
        - D
    )

    plt.plot(S,payoff); plt.axhline(0); plt.show()
    return K4-D, K1+D, kappa-D, D


# Example
long_call_condor(90,100,110,120, 12,7,3,1)

def long_put_condor(K1, K2, K3, K4, P1, P2, P3, P4, points=200):
    """
    ---------------------------------------------
    Long Put Condor
    Long  OTM put at K1
    Short OTM put at K2
    Short ITM put at K3
    Long  ITM put at K4
    Strikes equidistant
    Net debit strategy
    Trader outlook: neutral
    Capital gain strategy
    ---------------------------------------------

    Net debit D = P1 - P2 - P3 + P4
    Payoff = (K1-S)+ - (K2-S)+ - (K3-S)+ + (K4-S)+ - D
    """

    kappa = K2 - K1
    D = P1 - P2 - P3 + P4
    S = np.linspace(0.5*K1, 1.5*K4, points)

    payoff = (
        np.maximum(K1-S,0)
        - np.maximum(K2-S,0)
        - np.maximum(K3-S,0)
        + np.maximum(K4-S,0)
        - D
    )

    plt.plot(S,payoff); plt.axhline(0); plt.show()
    return K4-D, K1+D, kappa-D, D


# Example
long_put_condor(90,100,110,120, 1,3,7,12)

def short_call_condor(K1, K2, K3, K4, C1, C2, C3, C4, points=200):
    """
    ---------------------------------------------
    Short Call Condor
    Short ITM call at K1
    Long  ITM call at K2
    Long  OTM call at K3
    Short OTM call at K4
    Strikes equidistant
    Net credit strategy
    Trader outlook: neutral
    Capital gain strategy
    ---------------------------------------------

    Net credit C = C2 + C3 - C1 - C4
    Payoff = (S-K2)+ + (S-K3)+ - (S-K1)+ - (S-K4)+ + C
    """

    kappa = K2 - K1
    C = C2 + C3 - C1 - C4
    S = np.linspace(0.5*K1, 1.5*K4, points)

    payoff = (
        np.maximum(S-K2,0)
        + np.maximum(S-K3,0)
        - np.maximum(S-K1,0)
        - np.maximum(S-K4,0)
        + C
    )

    plt.plot(S,payoff); plt.axhline(0); plt.show()
    return K4-C, K1+C, C, kappa-C


# Example
short_call_condor(90,100,110,120, 12,7,3,1)

def short_put_condor(K1, K2, K3, K4, P1, P2, P3, P4, points=200):
    """
    ---------------------------------------------
    Short Put Condor
    Short OTM put at K1
    Long  OTM put at K2
    Long  ITM put at K3
    Short ITM put at K4
    Strikes equidistant
    Net credit strategy
    Trader outlook: neutral
    Capital gain strategy
    ---------------------------------------------

    Net credit C = P2 + P3 - P1 - P4
    Payoff = (K2-S)+ + (K3-S)+ - (K1-S)+ - (K4-S)+ + C
    """

    kappa = K2 - K1
    C = P2 + P3 - P1 - P4
    S = np.linspace(0.5*K1, 1.5*K4, points)

    payoff = (
        np.maximum(K2-S,0)
        + np.maximum(K3-S,0)
        - np.maximum(K1-S,0)
        - np.maximum(K4-S,0)
        + C
    )

    plt.plot(S,payoff); plt.axhline(0); plt.show()
    return K4-C, K1+C, C, kappa-C


# Example
short_put_condor(90,100,110,120, 1,3,7,12)

def long_iron_condor(K1, K2, K3, K4, P1, P2, C3, C4, points=200):
    """
    ---------------------------------------------
    Long Iron Condor
    Long  OTM put at K1
    Short OTM put at K2
    Short OTM call at K3
    Long  OTM call at K4
    Strikes equidistant
    Net credit strategy
    Trader outlook: neutral
    Income strategy
    ---------------------------------------------

    Net credit C = P2 + C3 - P1 - C4
    """

    kappa = K2 - K1
    C = P2 + C3 - P1 - C4
    S = np.linspace(0.5*K1, 1.5*K4, points)

    payoff = (
        np.maximum(K1-S,0)
        + np.maximum(S-K4,0)
        - np.maximum(K2-S,0)
        - np.maximum(S-K3,0)
        + C
    )

    plt.plot(S,payoff); plt.axhline(0); plt.show()
    return K3+C, K2-C, C, kappa-C


# Example
long_iron_condor(90,100,110,120, 1,3,3,1)

def short_iron_condor(K1, K2, K3, K4, P1, P2, C3, C4, points=200):
    """
    ---------------------------------------------
    Short Iron Condor
    Short OTM put at K1
    Long  OTM put at K2
    Long  OTM call at K3
    Short OTM call at K4
    Strikes equidistant
    Net debit strategy
    Trader outlook: neutral
    Capital gain strategy
    ---------------------------------------------

    Net debit D = P1 + C4 - P2 - C3
    """

    kappa = K2 - K1
    D = P1 + C4 - P2 - C3
    S = np.linspace(0.5*K1, 1.5*K4, points)

    payoff = (
        np.maximum(K2-S,0)
        + np.maximum(S-K3,0)
        - np.maximum(K1-S,0)
        - np.maximum(S-K4,0)
        - D
    )

    plt.plot(S,payoff); plt.axhline(0); plt.show()
    return K3+D, K2-D, kappa-D, D


# Example
short_iron_condor(90,100,110,120, 1,3,3,1)

def long_box(K1, K2, D):
    """
    ---------------------------------------------
    Long Box Spread
    Long  ITM put at K1
    Short OTM put at K2
    Long  ITM call at K2
    Short OTM call at K1
    Riskless payoff at expiry
    Trader outlook: neutral
    Capital gain strategy
    ---------------------------------------------

    Payoff = K1 - K2 - D
    """

    return K1 - K2 - D


# Example
long_box(110, 100, 2)

def collar_strategy(S0, K_put, K_call, P_price, C_price, H=0,
                    S_min=0.5, S_max=1.5, points=200):
    """
    Collar (Fence) Strategy
    Buy stock S0
    Buy OTM put at K_put
    Sell OTM call at K_call
    H = any additional cost/premium adjustments
    """

    # Stock price range
    S = np.linspace(S_min * S0, S_max * S0, points)

    # Payoff from stock
    stock_payoff = S - S0

    # Payoff from long put
    put_payoff = np.maximum(K_put - S, 0) - P_price

    # Payoff from short call
    call_payoff = -np.maximum(S - K_call, 0) + C_price

    # Total payoff
    payoff = stock_payoff + put_payoff + call_payoff - H

    # Max profit and loss
    P_max = K_call - S0 - H + C_price - P_price
    L_max = S0 - K_put + H + P_price - C_price

    # Break-even points (approximate)
    S_up = K_call + P_max
    S_down = K_put - L_max

    print("Approx. S_up:", S_up)
    print("Approx. S_down:", S_down)
    print("Max profit:", P_max)
    print("Max loss:", L_max)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Collar Strategy")
    plt.axhline(0, linewidth=1)
    plt.axvline(S_up, linestyle="--")
    plt.axvline(S_down, linestyle="--")
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Collar Strategy Payoff at Expiry")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, S_up, S_down, P_max, L_max

# Example usage
collar_strategy(S0=100, K_put=90, K_call=110, P_price=2.5, C_price=3.0)

def bearish_long_seagull(K1, K2, K3, P_price, C_ATM_price, C_OTM_price,
                         S_min=0.5, S_max=1.5, points=200, H=0):
    """
    Bearish Long Seagull Spread
    Long OTM put at K1
    Short ATM call at K2
    Long OTM call at K3
    """
    S = np.linspace(S_min * K1, S_max * K3, points)

    payoff = (
        np.maximum(K1 - S, 0)        # Long put
        - np.maximum(S - K2, 0)      # Short call ATM
        + np.maximum(S - K3, 0)      # Long call OTM
        - H
        - P_price + C_ATM_price - C_OTM_price  # premiums adjustment
    )

    P_max = K1 - H
    L_max = K3 - K2 + H

    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Bearish Long Seagull")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Bearish Long Seagull Spread Payoff")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, P_max, L_max

# Example 
bearish_long_seagull(K1=90, K2=100, K3=110, P_price=2.5, C_ATM_price=3.0, C_OTM_price=1.5)


def bullish_long_seagull(K1, K2, K3, P_OTM_price, P_ATM_price, C_OTM_price,
                         S_min=0.5, S_max=1.5, points=200, H=0):
    """
    Bullish Long Seagull Spread
    Long OTM put at K1
    Short ATM put at K2
    Long OTM call at K3
    """
    S = np.linspace(S_min * K1, S_max * K3, points)

    payoff = (
        np.maximum(K1 - S, 0)        # Long put OTM
        - np.maximum(K2 - S, 0)      # Short ATM put
        + np.maximum(S - K3, 0)      # Long call OTM
        - H
        - P_OTM_price + P_ATM_price - C_OTM_price  # premiums adjustment
    )

    P_max = np.inf  # unlimited
    L_max = K2 - K1 + H

    print("Max profit:", P_max)
    print("Max loss:", L_max)

    plt.figure(figsize=(7, 4))
    plt.plot(S, payoff, label="Bullish Long Seagull")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Stock price at expiry")
    plt.ylabel("Payoff")
    plt.title("Bullish Long Seagull Spread Payoff")
    plt.grid(True)
    plt.legend()
    plt.show()

    return S, payoff, P_max, L_max


# Example

bullish_long_seagull(K1=90, K2=100, K3=110, P_OTM_price=2.5, P_ATM_price=3.0, C_OTM_price=1.5)

