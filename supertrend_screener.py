"""
Supertrend + 10 EMA Weekly Swing Screener
==========================================
Standalone dashboard project (independent from Stock_squeeze_screener),
but scans the SAME universe: all US-listed stocks from the NASDAQ Trader
symbol directory, with tickers.txt as a fallback safety net.

Strategy (weekly timeframe):
  ENTRY    : Supertrend(10,3) crossed positive this week
  RE-ENTRY : Supertrend positive AND weekly candle crossed & closed above 10 EMA
  HOLD     : Supertrend positive AND close above 10 EMA (trend intact)
  EXIT     : Weekly candle closed below 10 EMA

Output: sorted shortlist + HTML dashboard (published to GitHub Pages, emailed).
"""

import io
import os
import smtplib
import ssl
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
ATR_PERIOD = 10
ATR_FACTOR = 3.0
EMA_LEN = 10
WEEKS_OF_DATA = "3y"            # weekly bars for stable ATR/EMA

# Universe (mirrors Stock_squeeze_screener defaults)
MAX_PRICE = 20.0                # None = no price cap
MIN_AVG_DOLLAR_VOL = 200_000    # liquidity floor (avg weekly $ vol / 5)
FALLBACK_FILE = "tickers.txt"   # used only if live universe fetch fails
MAX_TICKERS = 8000              # safety cap

# Download batching (Yahoo rate-limit friendly)
BATCH_SIZE = 200
BATCH_PAUSE = 2.0               # seconds between batches
MAX_RETRIES = 3

# Report caps (full universe = thousands of rows otherwise)
MAX_HOLD_ROWS = 30
MAX_EXIT_ROWS = 30

REPORT_FILE = "supertrend_report.html"
SIGNAL_ORDER = {"ENTRY": 0, "RE-ENTRY": 1, "HOLD": 2, "EXIT": 3, "BEARISH": 4}

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


# ----------------------------------------------------------
# Universe (same source as Stock_squeeze_screener)
# ----------------------------------------------------------
def fetch_universe() -> list[str]:
    tickers: set[str] = set()
    try:
        for url, sym_col in [(NASDAQ_LISTED, "Symbol"), (OTHER_LISTED, "ACT Symbol")]:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            df = df[df[sym_col].notna()]
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]
            if "ETF" in df.columns:
                df = df[df["ETF"] == "N"]
            for s in df[sym_col].astype(str):
                s = s.strip().upper()
                # skip units/warrants/preferreds/test rows
                if s and s.isalpha() and len(s) <= 5:
                    tickers.add(s)
        print(f"Universe: {len(tickers)} tickers from NASDAQ Trader")
    except Exception as e:
        print(f"Universe fetch failed ({e}) - using fallback file")
        if os.path.exists(FALLBACK_FILE):
            with open(FALLBACK_FILE) as f:
                tickers = {t.strip().upper() for t in f
                           if t.strip() and not t.startswith("#")}
    return sorted(tickers)[:MAX_TICKERS]


# ----------------------------------------------------------
# Indicators
# ----------------------------------------------------------
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()  # RMA, matches TradingView


def supertrend(df: pd.DataFrame, period: int = ATR_PERIOD, factor: float = ATR_FACTOR):
    """Returns (st_line, direction) with direction: 1 = bullish, -1 = bearish."""
    _atr = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = (hl2 + factor * _atr).to_numpy()
    lower = (hl2 - factor * _atr).to_numpy()
    close = df["Close"].to_numpy()

    n = len(df)
    st = np.full(n, np.nan)
    direction = np.full(n, -1, dtype=int)
    f_upper, f_lower = upper.copy(), lower.copy()

    for i in range(1, n):
        f_upper[i] = upper[i] if (upper[i] < f_upper[i-1] or close[i-1] > f_upper[i-1]) else f_upper[i-1]
        f_lower[i] = lower[i] if (lower[i] > f_lower[i-1] or close[i-1] < f_lower[i-1]) else f_lower[i-1]
        if direction[i-1] == -1:
            direction[i] = 1 if close[i] > f_upper[i] else -1
        else:
            direction[i] = -1 if close[i] < f_lower[i] else 1
        st[i] = f_lower[i] if direction[i] == 1 else f_upper[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


# ----------------------------------------------------------
# Signal classification
# ----------------------------------------------------------
def classify(df: pd.DataFrame) -> dict | None:
    if len(df) < ATR_PERIOD + 15:
        return None

    df = df.copy()
    close_now = float(df["Close"].iloc[-1])

    # Universe filters (mirrors squeeze screener: price cap + liquidity floor)
    if MAX_PRICE is not None and close_now > MAX_PRICE:
        return None
    avg_wk_dollar_vol = float((df["Close"] * df["Volume"]).iloc[-11:-1].mean())
    if avg_wk_dollar_vol / 5 < MIN_AVG_DOLLAR_VOL:   # rough daily equivalent
        return None

    df["EMA10"] = df["Close"].ewm(span=EMA_LEN, adjust=False).mean()
    st_line, st_dir = supertrend(df)
    df["ST"], df["ST_DIR"] = st_line, st_dir

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
    elif st_bull:
        signal = "EXIT"
    else:
        signal = "BEARISH"

    ema_dist = (c["Close"] - c["EMA10"]) / c["EMA10"] * 100
    st_dist = (c["Close"] - c["ST"]) / c["ST"] * 100 if not np.isnan(c["ST"]) else np.nan
    vol_avg = df["Volume"].iloc[-11:-1].mean()
    vol_ratio = c["Volume"] / vol_avg if vol_avg > 0 else np.nan

    return {
        "Signal": signal,
        "Close": round(close_now, 2),
        "EMA10": round(float(c["EMA10"]), 2),
        "EMA_Dist%": round(float(ema_dist), 2),
        "ST_Dist%": round(float(st_dist), 2) if not np.isnan(st_dist) else None,
        "Vol_x_Avg": round(float(vol_ratio), 2) if not np.isnan(vol_ratio) else None,
        "Week": df.index[-1].strftime("%d %b %Y"),
    }


# ----------------------------------------------------------
# Scan (batched, retry/backoff - same reliability pattern as squeeze screener)
# ----------------------------------------------------------
def scan(tickers: list[str]) -> tuple[pd.DataFrame, int]:
    rows, scanned = [], 0
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for bi, batch in enumerate(batches, 1):
        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = yf.download(batch, period=WEEKS_OF_DATA, interval="1wk",
                                   group_by="ticker", auto_adjust=True,
                                   threads=True, progress=False)
                break
            except Exception as e:
                print(f"batch {bi} attempt {attempt} failed: {e}")
                time.sleep(5 * attempt)
        if data is None or data.empty:
            print(f"batch {bi}: no data, skipping")
            continue
        for t in batch:
            try:
                df = data[t].dropna() if len(batch) > 1 else data.dropna()
                if df.empty:
                    continue
                scanned += 1
                res = classify(df)
                if res:
                    rows.append({"Ticker": t, **res})
            except Exception:
                continue
        print(f"batch {bi}/{len(batches)} done - {len(rows)} matches so far")
        time.sleep(BATCH_PAUSE)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, scanned
    out["_order"] = out["Signal"].map(SIGNAL_ORDER)
    out = out.sort_values(["_order", "Vol_x_Avg"], ascending=[True, False]).drop(columns="_order")
    return out.reset_index(drop=True), scanned


def trim_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """All ENTRY/RE-ENTRY, top HOLD/EXIT by volume; BEARISH counted but not listed."""
    if df.empty:
        return df
    parts = [
        df[df["Signal"] == "ENTRY"],
        df[df["Signal"] == "RE-ENTRY"],
        df[df["Signal"] == "HOLD"].head(MAX_HOLD_ROWS),
        df[df["Signal"] == "EXIT"].head(MAX_EXIT_ROWS),
    ]
    return pd.concat(parts).reset_index(drop=True)


# ----------------------------------------------------------
# HTML dashboard
# ----------------------------------------------------------
SIGNAL_COLORS = {
    "ENTRY": "#16a34a", "RE-ENTRY": "#0d9488",
    "HOLD": "#2563eb", "EXIT": "#dc2626", "BEARISH": "#6b7280",
}


SELECT_STYLE = ("padding:4px 8px;border:1px solid #cbd5e1;border-radius:6px;"
                "background:#fff;color:#334155;font-size:13px;")

VIEW_SCRIPT = """
<script>
function applyView() {
  var tbody = document.getElementById('rows');
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr[data-signal]'));
  var sig = document.getElementById('filterSignal').value;
  var field = document.getElementById('sortField').value;
  var asc = document.getElementById('sortOrder').value === 'asc';
  var shown = 0;
  rows.forEach(function (r) {
    var visible = (sig === 'ALL' || r.dataset.signal === sig);
    r.style.display = visible ? '' : 'none';
    if (visible) shown++;
  });
  rows.sort(function (a, b) {
    if (field === 'ticker') {
      var cmp = a.dataset.ticker.localeCompare(b.dataset.ticker);
      return asc ? cmp : -cmp;
    }
    var va = parseFloat(a.dataset[field]);
    var vb = parseFloat(b.dataset[field]);
    if (isNaN(va)) va = asc ? Infinity : -Infinity;
    if (isNaN(vb)) vb = asc ? Infinity : -Infinity;
    return asc ? va - vb : vb - va;
  });
  rows.forEach(function (r) { tbody.appendChild(r); });
  var counter = document.getElementById('rowCount');
  if (counter) counter.textContent = shown + ' of ' + rows.length + ' rows';
}
</script>"""


def build_controls() -> str:
    signal_opts = "".join(f'<option value="{s}">{s}</option>' for s in SIGNAL_ORDER)
    return f"""
    <div style="padding:0 24px 16px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;
                font-size:13px;color:#334155;">
      <label>Filter:
        <select id="filterSignal" onchange="applyView()" style="{SELECT_STYLE}">
          <option value="ALL">All signals</option>{signal_opts}
        </select>
      </label>
      <label>Sort by:
        <select id="sortField" onchange="applyView()" style="{SELECT_STYLE}">
          <option value="idx">Signal priority (default)</option>
          <option value="ticker">Ticker</option>
          <option value="close">Close</option>
          <option value="emadist">vs EMA %</option>
          <option value="stdist">vs ST %</option>
          <option value="vol">Vol vs 10wk avg</option>
        </select>
      </label>
      <label>Order:
        <select id="sortOrder" onchange="applyView()" style="{SELECT_STYLE}">
          <option value="asc">Ascending</option>
          <option value="desc">Descending</option>
        </select>
      </label>
      <span id="rowCount" style="color:#94a3b8;"></span>
    </div>"""


def build_html(full: pd.DataFrame, table: pd.DataFrame, scanned: int) -> str:
    ts = datetime.now().strftime("%A %d %B %Y, %H:%M UTC")
    counts = full["Signal"].value_counts().to_dict() if not full.empty else {}
    badges = "".join(
        f'<span style="background:{SIGNAL_COLORS[s]};color:#fff;border-radius:12px;'
        f'padding:4px 12px;margin-right:8px;font-size:13px;">{s}: {counts.get(s, 0)}</span>'
        for s in SIGNAL_ORDER
    )

    if table.empty:
        body_rows = '<tr><td colspan="8" style="padding:16px;text-align:center;">No results</td></tr>'
    else:
        body_rows = ""
        for i, (_, r) in enumerate(table.iterrows()):
            col = SIGNAL_COLORS[r["Signal"]]
            num = lambda v: "" if v is None else v
            body_rows += f"""
            <tr style="border-bottom:1px solid #e5e7eb;" data-signal="{r['Signal']}"
                data-ticker="{r['Ticker']}" data-idx="{i}" data-close="{r['Close']}"
                data-emadist="{r['EMA_Dist%']}" data-stdist="{num(r['ST_Dist%'])}"
                data-vol="{num(r['Vol_x_Avg'])}">
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
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supertrend + 10EMA Weekly Dashboard</title></head>
<body style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f8fafc;margin:0;padding:24px;">
  <div style="max-width:900px;margin:auto;background:#fff;border-radius:12px;
              box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">
    <div style="background:#0f172a;color:#fff;padding:20px 24px;">
      <h1 style="margin:0;font-size:20px;">Supertrend(10,3) + 10 EMA - Weekly Swing Dashboard</h1>
      <p style="margin:6px 0 0;color:#94a3b8;font-size:13px;">Generated {ts} - Weekly candles -
         {scanned} tickers scanned (full US universe, max ${MAX_PRICE} + liquidity filter)</p>
    </div>
    <div style="padding:16px 24px;">{badges}</div>
    {build_controls()}
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr style="background:#f1f5f9;text-align:left;color:#334155;">
          <th style="padding:10px 12px;">Ticker</th><th style="padding:10px 12px;">Signal</th>
          <th style="padding:10px 12px;">Close</th><th style="padding:10px 12px;">10 EMA</th>
          <th style="padding:10px 12px;">vs EMA</th><th style="padding:10px 12px;">vs ST</th>
          <th style="padding:10px 12px;">Vol vs 10wk</th><th style="padding:10px 12px;">Week</th>
        </tr>
      </thead>
      <tbody id="rows">{body_rows}</tbody>
    </table>
    <p style="padding:16px 24px;color:#94a3b8;font-size:12px;">
      ENTRY = Supertrend flipped positive - RE-ENTRY = ST positive &amp; close crossed above 10EMA -
      EXIT = weekly close below 10EMA (top {MAX_EXIT_ROWS} shown) - HOLD capped at {MAX_HOLD_ROWS} rows -
      BEARISH counted but not listed. Educational use only - not financial advice.
    </p>
  </div>
  {VIEW_SCRIPT}
  <script>applyView();</script>
</body></html>"""


# ----------------------------------------------------------
# Email
# ----------------------------------------------------------
def send_email(html: str):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("EMAIL_RECIPIENT", sender)
    if not sender or not password:
        print("Email secrets not set - skipping email, report saved locally.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Supertrend+10EMA Weekly Dashboard - {datetime.now():%d %b %Y}"
    msg["From"], msg["To"] = sender, recipient
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, password)
        s.sendmail(sender, recipient, msg.as_string())
    print(f"Report emailed to {recipient}")


# ----------------------------------------------------------
if __name__ == "__main__":
    tickers = fetch_universe()
    print(f"Scanning {len(tickers)} tickers on weekly timeframe...")
    results, scanned = scan(tickers)
    table = trim_for_report(results)
    print(table.to_string(index=False) if not table.empty else "No signals.")
    html = build_html(results, table, scanned)
    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"Saved {REPORT_FILE}")
    send_email(html)
