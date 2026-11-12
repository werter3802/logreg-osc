import json, os
import numpy as np, pandas as pd, requests, yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

TOKEN, CHAT_ID = os.environ["TG_TOKEN"], os.environ["TG_CHAT_ID"]

SYMBOLS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
           "XAUUSD": "GC=F", "US100": "NQ=F", "US500": "ES=F", "BTCUSD": "BTC-USD"}
TIMEFRAMES = {"5m": "7d", "15m": "20d"}   # interwał: ile historii pobrać
LR_LEN, LR_LAG = 100, 3      # ustaw jak we wskaźniku LogReg
AO_LEN, AO_SIG = 50, 9       # ustaw jak w Andean Oscillator
MAX_WAIT_BARS = 30           # Andean ma się zgrać w ciągu tylu świec od zmiany koloru (None = bez limitu)
MAX_ALERT_AGE = 6            # alert tylko, jeśli zgranie było w ostatnich N zamkniętych świecach
STATE_FILE = "state.json"

def log_reg_line(close, n):
    y = np.log(close)
    out = np.full(len(y), np.nan)
    if len(y) < n:
        return out
    w = sliding_window_view(y, n)
    x = np.arange(n); xm = x.mean()
    slope = ((w - w.mean(1, keepdims=True)) * (x - xm)).sum(1) / ((x - xm) ** 2).sum()
    out[n - 1:] = np.exp(w.mean(1) + slope * (n - 1 - xm))
    return out

def andean(o, c, length, sig_len):
    a = 2 / (length + 1)
    n = len(c)
    up1 = np.empty(n); up2 = np.empty(n); dn1 = np.empty(n); dn2 = np.empty(n)
    up1[0] = dn1[0] = c[0]; up2[0] = dn2[0] = c[0] ** 2
    for i in range(1, n):
        up1[i] = max(c[i], o[i], up1[i-1] - (up1[i-1] - c[i]) * a)
        up2[i] = max(c[i]**2, o[i]**2, up2[i-1] - (up2[i-1] - c[i]**2) * a)
        dn1[i] = min(c[i], o[i], dn1[i-1] + (c[i] - dn1[i-1]) * a)
        dn2[i] = min(c[i]**2, o[i]**2, dn2[i-1] + (c[i]**2 - dn2[i-1]) * a)
    bull = np.sqrt(np.maximum(dn2 - dn1**2, 0))   # zielona
    bear = np.sqrt(np.maximum(up2 - up1**2, 0))   # czerwona
    sig = pd.Series(np.maximum(bull, bear)).ewm(span=sig_len, adjust=False).mean().to_numpy()  # pomarańczowa
    return bull, sig, bear

def get_df(ticker, tf):
    df = yf.Ticker(ticker).history(period=TIMEFRAMES[tf], interval=tf)
    return df if len(df) >= LR_LEN + 200 else None

def check(df):
    df = df.iloc[:-1]                                    # odrzuć niezamkniętą świecę
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    n = len(c)

    # kolor środkowej linii logreg: +1 zielony, -1 czerwony (zero przenosi poprzedni kolor)
    reg = log_reg_line(c, LR_LEN)
    t = pd.Series(np.sign(reg - np.roll(reg, LR_LAG)))
    t.iloc[:LR_LEN + LR_LAG] = np.nan
    t = t.replace(0, np.nan).ffill().to_numpy()
    if np.isnan(t[-1]):
        return None
    direction = t[-1]

    # ostatnia zmiana koloru; wszystko przed nią jest anulowane
    valid = ~np.isnan(t)
    changes = np.where(valid[1:] & valid[:-1] & (t[1:] != t[:-1]))[0] + 1
    if len(changes) == 0:
        return None
    f = changes[-1]

    # pierwsze zgranie Andean po zmianie koloru (kolor logreg nadal ten sam)
    bull, sig, bear = andean(o, c, AO_LEN, AO_SIG)
    aligned = (bull > sig) & (sig > bear) if direction == 1 else (bear > sig) & (sig > bull)
    end = n if MAX_WAIT_BARS is None else min(n, f + MAX_WAIT_BARS + 1)
    hits = np.where(aligned[f:end])[0]
    if len(hits) == 0:
        return None
    j = f + hits[0]
    if n - 1 - j >= MAX_ALERT_AGE:
        return None
    return ("LONG 🟢" if direction == 1 else "SHORT 🔴"), df.index[f], df.index[j]

def send(text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": text}, timeout=15)

def main():
    state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    for name, ticker in SYMBOLS.items():
        for tf in TIMEFRAMES:
            try:
                df = get_df(ticker, tf)
                if df is None:
                    print(name, tf, "brak danych"); continue
                res = check(df)
            except Exception as e:
                print(name, tf, "błąd:", e); continue
            if not res:
                continue
            side, t_flip, t_hit = res
            key = f"{name}-{tf}"
            if state.get(key) != str(t_flip):            # jeden alert na jedną zmianę koloru
                send(f"{name} {tf}: {side}\nLogReg zmienił kolor: {t_flip}\n"
                     f"Andean zgrany: {t_hit}\nKolor logreg nadal ten sam")
                state[key] = str(t_flip)
    json.dump(state, open(STATE_FILE, "w"))

if __name__ == "__main__":
    main()
