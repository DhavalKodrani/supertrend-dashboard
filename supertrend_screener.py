"""
Supertrend + 10 EMA Weekly Swing Screener
==========================================
Separate dashboard project (independent from Stock_squeeze_screener).

Strategy (weekly timeframe):
  ENTRY    : Supertrend(10,3) crossed positive this week
  RE-ENTRY : Supertrend positive AND weekly candle crossed & closed above 10 EMA
  HOLD     : Supertrend positive AND close above 10 EMA (trend intact)
  EXIT     : Weekly candle closed below 10 EMA

Output: sorted shortlist + HTML dashboard report (emailed via GitHub Actions).
"""

import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
ATR_PERIOD = 10
ATR_FACTOR = 3.0
EMA_LEN = 10
WEEKS_OF_DATA = "3y"          # enough weekly bars for stable ATR/EMA
TICKER_FILE = "tickers.txt"   # one ticker per line
REPORT_FILE = "supertrend_report.html"

SIGNAL_ORDER = {"ENTRY": 0, "RE-ENTRY": 1, "HOLD": 2, "EXIT": 3, "BEARISH": 4}


# ──────────────────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────────────────
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()  # RMA, matches TradingView


def supertrend(df: pd.DataFrame, period: int = ATR_PERIOD, factor: float = ATR_FACTOR):
    """Returns (st_line, direction) where direction: 1 = bullish, -1 = bearish."""
    _atr = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + factor * _atr
    lower = hl2 - factor * _atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    f_upper = upper.copy()
    f_lower = lower.copy()

    for i in range(1, len(df)):
        # Final bands
        if upper.iloc[i] < f_upper.iloc[i - 1] or df["Close"].iloc[i - 1] > f_upper.iloc[i - 1]:
            f_upper.iloc[i] = upper.iloc[i]
        else:
            f_upper.iloc[i] = f_upper.iloc[i - 1]

        if lower.iloc[i] > f_lower.iloc[i - 1] or df["Close"].iloc[i - 1] < f_lower.iloc[i - 1]:
            f_lower.iloc[i] = lower.iloc[i]
        else:
            f_lower.iloc[i] = f_lower.iloc[i - 1]

        # Direction
        prev_dir = direction.iloc[i - 1] if not np.isnan(direction.iloc[i - 1]) else -1
        if prev_dir == -1:
            direction.iloc[i] = 1 if df["Close"].iloc[i] > f_upper.iloc[i] else -1
        else:
            direction.iloc[i] = -1 if df["Close"].iloc[i] < f_lower.iloc[i] else 1

        st.iloc[i] = f_lower.iloc[i] if direction.iloc[i] == 1 else f_upper.iloc[i]

    return st, direction


# ──────────────────────────────────────────────────────────
# Signal classification
# ──────────────────────────────────────────────────────────
def classify(df: pd.DataFrame) -> dict | None:
    if len(df) < ATR_PERIOD + 15:
        return None

    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=EMA_LEN, adjust=False).mean()
    st_line, st_dir = supertrend(df)
    df["ST"] = st_line
    df["ST_DIR"] = st_dir

    c, p = df.iloc[-1], df.iloc[-2]

    st_bull = c["ST_DIR"] == 1
    st_flip = c["ST_DIR"] == 1 and p["ST_DIR"] == -1
    above_ema = c["Close"] > c["EMA10"]
    crossed_ema = above_ema and p["Close"] <= p["EMA10"]

    if st_flip:
        signal = "ENTRY"
    elif st_bull and crossed_ema:
        signal = "RE-ENTRY"
    elif st_bull and above_ema:
        signal = "HOLD"
    elif st_bull and not above_ema:
        signal = "EXIT"
    else:
        signal = "BEARISH"

    ema_dist = (c["Close"] - c["EMA10"]) / c["EMA10"] * 100
    st_dist = (c["Close"] - c["ST"]) / c["ST"] * 100 if not np.isnan(c["ST"]) else np.nan
    vol_ratio = c["Volume"] / df["Volume"].iloc[-11:-1].mean() if df["Volume"].iloc[-11:-1].mean() > 0 else np.nan

    return {
        "Signal": signal,
        "Close": round(float(c["Close"]), 2),
        "EMA10": round(float(c["EMA10"]), 2),
        "EMA_Dist%": round(float(ema_dist), 2),
        "ST_Dist%": round(float(st_dist), 2) if not np.isnan(st_dist) else None,
        "Vol_x_Avg": round(float(vol_ratio), 2) if not np.isnan(vol_ratio) else None,
        "Week": df.index[-1].strftime("%d %b %Y"),
    }


# ──────────────────────────────────────────────────────────
# Scan
# ──────────────────────────────────────────────────────────
def load_tickers() -> list[str]:
    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE) as f:
            return [t.strip().upper() for t in f if t.strip() and not t.startswith("#")]
    return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]  # fallback


def scan(tickers: list[str]) -> pd.DataFrame:
    rows = []
    data = yf.download(tickers, period=WEEKS_OF_DATA, interval="1wk",
                       group_by="ticker", auto_adjust=True, threads=True, progress=False)
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers) > 1 else data.dropna()
            res = classify(df)
            if res:
                rows.append({"Ticker": t, **res})
        except Exception as e:
            print(f"  skip {t}: {e}")
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_order"] = out["Signal"].map(SIGNAL_ORDER)
    # Actionable first, then strongest momentum (volume ratio) within each group
    out = out.sort_values(["_order", "Vol_x_Avg"], ascending=[True, False]).drop(columns="_order")
    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# HTML dashboard (same visual language as squeeze screener, own page)
# ──────────────────────────────────────────────────────────
SIGNAL_COLORS = {
    "ENTRY": "#16a34a", "RE-ENTRY": "#0d9488",
    "HOLD": "#2563eb", "EXIT": "#dc2626", "BEARISH": "#6b7280",
}


def build_html(df: pd.DataFrame) -> str:
    ts = datetime.now().strftime("%A %d %B %Y, %H:%M")
    counts = df["Signal"].value_counts().to_dict() if not df.empty else {}
    badges = "".join(
        f'<span style="background:{SIGNAL_COLORS[s]};color:#fff;border-radius:12px;'
        f'padding:4px 12px;margin-right:8px;font-size:13px;">{s}: {counts.get(s, 0)}</span>'
        for s in SIGNAL_ORDER
    )

    if df.empty:
        body_rows = '<tr><td colspan="8" style="padding:16px;text-align:center;">No results</td></tr>'
    else:
        body_rows = ""
        for _, r in df.iterrows():
            col = SIGNAL_COLORS[r["Signal"]]
            body_rows += f"""
            <tr style="border-bottom:1px solid #e5e7eb;">
              <td style="padding:8px 12px;font-weight:600;">{r['Ticker']}</td>
              <td style="padding:8px 12px;"><span style="background:{col};color:#fff;
                  border-radius:6px;padding:2px 10px;font-size:12px;">{r['Signal']}</span></td>
              <td style="padding:8px 12px;">{r['Close']}</td>
              <td style="padding:8px 12px;">{r['EMA10']}</td>
              <td style="padding:8px 12px;">{r['EMA_Dist%']}%</td>
              <td style="padding:8px 12px;">{r['ST_Dist%']}%</td>
              <td style="padding:8px 12px;">{r['Vol_x_Avg']}x</td>
              <td style="padding:8px 12px;color:#6b7280;">{r['Week']}</td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Supertrend + 10EMA Weekly Dashboard</title></head>
<body style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f8fafc;margin:0;padding:24px;">
  <div style="max-width:900px;margin:auto;background:#fff;border-radius:12px;
              box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">
    <div style="background:#0f172a;color:#fff;padding:20px 24px;">
      <h1 style="margin:0;font-size:20px;">📈 Supertrend(10,3) + 10 EMA — Weekly Swing Dashboard</h1>
      <p style="margin:6px 0 0;color:#94a3b8;font-size:13px;">Generated {ts} · Weekly candles</p>
    </div>
    <div style="padding:16px 24px;">{badges}</div>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr style="background:#f1f5f9;text-align:left;color:#334155;">
          <th style="padding:10px 12px;">Ticker</th><th style="padding:10px 12px;">Signal</th>
          <th style="padding:10px 12px;">Close</th><th style="padding:10px 12px;">10 EMA</th>
          <th style="padding:10px 12px;">vs EMA</th><th style="padding:10px 12px;">vs ST</th>
          <th style="padding:10px 12px;">Vol vs 10wk</th><th style="padding:10px 12px;">Week</th>
        </tr>
      </thead>
      <tbody>{body_rows}</tbody>
    </table>
    <p style="padding:16px 24px;color:#94a3b8;font-size:12px;">
      ENTRY = Supertrend flipped positive · RE-ENTRY = ST positive &amp; close crossed above 10EMA ·
      EXIT = weekly close below 10EMA. Educational use only — not financial advice.
    </p>
  </div>
</body></html>"""


# ──────────────────────────────────────────────────────────
# Email (same env-secret pattern as Stock_squeeze_screener)
# ──────────────────────────────────────────────────────────
def send_email(html: str):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("EMAIL_RECIPIENT", sender)
    if not sender or not password:
        print("Email secrets not set — skipping email, report saved locally.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Supertrend+10EMA Weekly Dashboard — {datetime.now():%d %b %Y}"
    msg["From"], msg["To"] = sender, recipient
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, password)
        s.sendmail(sender, recipient, msg.as_string())
    print(f"Report emailed to {recipient}")


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    tickers = load_tickers()
    print(f"Scanning {len(tickers)} tickers on weekly timeframe...")
    results = scan(tickers)
    print(results.to_string(index=False) if not results.empty else "No signals.")
    html = build_html(results)
    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"Saved {REPORT_FILE}")
    send_email(html)
