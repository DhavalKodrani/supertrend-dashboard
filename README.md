# Supertrend + 10 EMA Weekly Dashboard

Standalone swing-trading screener (separate project from Stock_squeeze_screener).

**Strategy (weekly candles):**
- ENTRY — Supertrend(10,3) crossed positive this week
- RE-ENTRY — Supertrend positive AND close crossed above 10 EMA
- HOLD — Supertrend positive, close above 10 EMA
- EXIT — weekly close below 10 EMA
- BEARISH — Supertrend negative

**Output:** sorted shortlist (actionable signals first, ranked by volume vs 10-week average) as an HTML dashboard, saved as `supertrend_report.html` and emailed.

## Setup
1. Edit `tickers.txt` with your universe.
2. Add repo secrets: `EMAIL_SENDER`, `EMAIL_PASSWORD` (Gmail app password), `EMAIL_RECIPIENT`.
3. Runs every Saturday 07:00 UTC via GitHub Actions, or trigger manually from the Actions tab.

## Local run
```
pip install -r requirements.txt
python supertrend_screener.py
```
