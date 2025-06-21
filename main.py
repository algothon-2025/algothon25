import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def cs_rank(x):
    """Cross-sectional rank, returns values between -0.5 and 0.5"""
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    return (ranks - np.mean(ranks)) / len(x)

def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    if nt < 10:
        return np.zeros(nInst, dtype=int)

    prices = prcSoFar[:, -1]
    max_dollar = 10000

    # --- Simple Momentum Strategy ---
    # Use 10-day momentum (shorter for more responsiveness)
    lookback = 10
    momentum = (prices - prcSoFar[:, -lookback]) / prcSoFar[:, -lookback]
    
    # Simple signal: positive momentum = long, negative = short
    signal = momentum
    
    # --- Volatility scaling ---
    recent_returns = np.diff(prcSoFar[:, -10:], axis=1) / prcSoFar[:, -10:-1]
    vol = np.std(recent_returns, axis=1) + 1e-6
    inv_vol = 1 / vol
    inv_vol = inv_vol / np.mean(inv_vol)
    signal = signal * inv_vol
    
    # --- Lower threshold for more trades ---
    threshold = 0.01  # 1% momentum threshold (lower than before)
    signal[np.abs(signal) < threshold] = 0
    
    # --- Dollar neutrality ---
    signal = signal - np.mean(signal)
    
    # --- Moderate position sizing ---
    max_shares = np.floor(max_dollar / prices) * 0.5  # Use 50% of max (increased from 30%)
    raw_pos = signal * max_shares
    positions = np.clip(np.round(raw_pos), -max_shares, max_shares).astype(int)
    
    return positions
