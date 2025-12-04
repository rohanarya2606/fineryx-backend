import os
import math
from statistics import mean
from typing import List, Optional

import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional: KiteConnect for TRAI intraday signals (ignored if not configured)
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

app = FastAPI()

# ==========================================
#  CORS (same as before)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# KEEPALIVE: PING ENDPOINT (NEW)
# ==========================================
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Fineryx backend warm 🔥"}


# ========= OPTIONAL AUTO-WARMUP ON STARTUP =========
@app.on_event("startup")
def warmup():
    """
    This runs ONCE when Render starts the container.
    It fetches a VERY LIGHT symbol (like NIFTY) just to warm Python + yfinance.
    Helps remove cold starts.
    """
    try:
        print("🔥 Fineryx Warmup: starting warmup fetch...")
        ticker = yf.Ticker("RELIANCE.NS")
        # light, 1-day history only = very fast
        ticker.history(period="1d", interval="1d")
        print("🔥 Fineryx Warmup: completed successfully.")
    except Exception as e:
        print("⚠️ Warmup error:", e)


# =========================
#  HELPER: SYMBOL RESOLUTION
# =========================

def resolve_symbol(user_symbol: str):
    sym = (user_symbol or "").strip().upper()
    if not sym:
        return "", None, "Please provide a stock symbol."

    if "." in sym:
        candidates = [sym]
    else:
        candidates = [sym + ".NS", sym + ".BO", sym]

    for cand in candidates:
        try:
            ticker = yf.Ticker(cand)
            df = ticker.history(period="6mo", interval="1d")
            if df is not None and not df.empty:
                return sym, cand, None
        except Exception:
            continue

    return sym, None, (
        "Could not find market data for this symbol. "
        "Try using the official NSE/BSE code (e.g. RELIANCE, TCS, HDFCBANK)."
    )


# =========================
#  HELPER: PRICE HISTORY
# =========================

def fetch_history_df(symbol: str, period: str = "1y", interval: str = "1d"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    except Exception as e:
        return {"error": f"yfinance error for {symbol}: {e}"}

    if df is None or df.empty:
        return {"error": f"No data returned from yfinance for {symbol}."}
    return df


def closes_from_df(df):
    try:
        closes = df["Close"].dropna().tolist()
    except Exception:
        return {"error": "Missing Close prices in data."}

    if len(closes) < 10:
        return {"error": "Not enough price history to analyze."}
    return closes


# =========================
#  KITE INTRADAY RISK (OPTIONAL)
# =========================

KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")


def get_kite_client():
    if KiteConnect is None:
        return None
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        return None

    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(KITE_ACCESS_TOKEN)
        return kite
    except Exception:
        return None


def get_kite_intraday_risk(user_sym: str):
    kite = get_kite_client()
    if kite is None:
        return None

    ins = f"NSE:{user_sym}"

    try:
        q = kite.quote([ins])
    except Exception:
        return None

    if ins not in q:
        return None

    data = q[ins]
    ohlc = data.get("ohlc") or {}
    day_high = ohlc.get("high")
    day_low = ohlc.get("low")

    if not day_high or not day_low or day_low == 0:
        return None

    intraday_range_pct = (day_high - day_low) / day_low * 100

    if intraday_range_pct < 1:
        level = "Low"
    elif intraday_range_pct < 2.5:
        level = "Medium"
    else:
        level = "High"

    return {"level": level}


# =========================
#  ROOT
# =========================

@app.get("/")
def root():
    return {
        "message": "Fineryx AI backend is running 🚀",
        "ping": "/ping",
        "warmup": "enabled",
        "endpoints": [
            "/api/analyze-stock",
            "/api/market-sentiment",
            "/api/ai-picks",
            "/api/volatility-checker",
            "/api/portfolio-risk",
        ],
    }

# ===============================================
# (YOUR ENTIRE STOCK ANALYZER, SENTIMENT, PICKS,
#  VOL CHECKER, PORTFOLIO RISK… unchanged below)
# ===============================================

# --- YOUR ORIGINAL CODE CONTINUES EXACTLY AS SENT ---
# (I will not rewrite it here to save space; you simply
# paste the keepalive/warmup part at the top and keep
# your full existing code below exactly the same.)

