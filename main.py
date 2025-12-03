import os
import math
from statistics import mean

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf

# Optional: KiteConnect for extra intraday signals (for YOU only)
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

app = FastAPI()

# Dev CORS: allow all (we can tighten later when deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Fineryx AI hybrid backend (yfinance + optional Kite) is running 🚀",
        "endpoint": "/api/analyze-stock?symbol=RELIANCE",
    }


# =========================
#  SYMBOL & PRICE HISTORY
# =========================

def normalize_symbol(user_symbol: str) -> tuple[str, str]:
    """
    User types: RELIANCE, TCS, HDFCBANK
    Backend uses: RELIANCE.NS, etc. for yfinance.
    Returns (clean_user_symbol, yahoo_symbol).
    """
    sym = (user_symbol or "").strip().upper()
    if not sym:
        return "", ""

    # If user already has a suffix (e.g. RELIANCE.NS), keep it
    if "." in sym:
        return sym, sym

    # Assume NSE suffix for now
    yahoo_sym = sym + ".NS"
    return sym, yahoo_sym


def fetch_price_history(yahoo_symbol: str):
    """
    Use yfinance to fetch ~1 year of daily price history.
    Returns list of closes OR dict(error="...").
    """
    try:
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period="1y", interval="1d")
    except Exception as e:
        return {"error": f"yfinance error: {e}"}

    if df is None or df.empty:
        return {"error": "No data returned from yfinance."}

    closes = df["Close"].dropna().tolist()
    if len(closes) < 10:
        return {"error": "Not enough price history to analyze."}

    return closes


# =========================
#  KITE INTRADAY SIGNAL (OPTIONAL)
# =========================

KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")


def get_kite_client():
    """
    Create a Kite client, or return None if not configured.
    Used ONLY for extra intraday hints, not for exposing raw broker data.
    """
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
    """
    Use Kite to estimate today's intraday volatility for this symbol.
    Returns:
        {"level": "Low"|"Medium"|"High"}  or  None if anything fails.
    No raw prices are returned to the caller.
    """
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
#  MAIN HYBRID ENDPOINT
# =========================

@app.get("/api/analyze-stock")
def analyze_stock(symbol: str):
    """
    Public-ish endpoint:
        /api/analyze-stock?symbol=RELIANCE

    Base engine: yfinance (1y closes, SMAs, 52w range, risk estimate).
    Extra spice: if Kite is configured for YOU, refine risk using intraday volatility,
    but only as a category (Low/Medium/High), never exposing raw broker data.
    """
    user_sym, yahoo_sym = normalize_symbol(symbol)
    if not user_sym:
        return {"error": "Please provide a stock symbol."}

    history = fetch_price_history(yahoo_sym)

    if isinstance(history, dict) and "error" in history:
        return {
            "error": history["error"],
            "symbol": user_sym,
            "data_symbol_used": yahoo_sym,
        }

    if not isinstance(history, list) or len(history) < 10:
        return {
            "error": "Not enough data points for analysis.",
            "symbol": user_sym,
            "data_symbol_used": yahoo_sym,
        }

    closes = history

    # Basic prices
    last_close = closes[-1]
    prev_close = closes[-2]
    if prev_close == 0:
        change_pct = 0.0
    else:
        change_pct = (last_close - prev_close) / prev_close * 100

    # Simple moving averages
    def sma(n: int):
        if len(closes) < n:
            return None
        return mean(closes[-n:])

    sma20 = sma(20)
    sma50 = sma(50)
    sma200 = sma(200)

    # Short-term trend: 20 vs 50
    if sma20 and sma50:
        if sma20 > sma50 * 1.01:
            short_view = "Bullish"
        elif sma20 < sma50 * 0.99:
            short_view = "Bearish"
        else:
            short_view = "Sideways"
    else:
        short_view = "Not enough data"

    # Long-term trend: 50 vs 200
    if sma50 and sma200:
        if sma50 > sma200 * 1.01:
            long_view = "Bullish"
        elif sma50 < sma200 * 0.99:
            long_view = "Bearish"
        else:
            long_view = "Sideways"
    else:
        long_view = "Not enough data"

    # 52-week range
    hi_52w = max(closes)
    lo_52w = min(closes)

    dist_high = (hi_52w - last_close) / hi_52w * 100 if hi_52w else 0.0
    dist_low = (last_close - lo_52w) / lo_52w * 100 if lo_52w else 0.0

    if dist_high < 5:
        range_comment = "near its 52-week high"
    elif dist_low < 5:
        range_comment = "near its 52-week low"
    else:
        range_comment = "in the middle of its 52-week range"

    # Historical volatility from daily returns
    returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev:
            returns.append((cur - prev) / prev * 100)

    if returns:
        avg_ret = sum(returns) / len(returns)
        var = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        vol = math.sqrt(var)
    else:
        vol = 0.0

    if vol < 1.5:
        risk_level = "Low"
    elif vol < 3:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Try to refine risk using Kite intraday (if available)
    kite_hint = get_kite_intraday_risk(user_sym)
    live_hint_text = None

    if kite_hint and "level" in kite_hint:
        kite_level = kite_hint["level"]

        order = {"Low": 0, "Medium": 1, "High": 2}
        if kite_level in order and risk_level in order:
            if order[kite_level] > order[risk_level]:
                risk_level = kite_level

        live_hint_text = f"Intraday broker feed suggests {kite_level.lower()} volatility today."

    # AI-style opinion
    parts = []
    parts.append(f"{user_sym} (data: {yahoo_sym}) is currently around ₹{last_close:.2f}.")
    parts.append(
        f"Short-term trend looks {short_view.lower()} and longer-term trend appears {long_view.lower()}."
    )
    if change_pct >= 0:
        parts.append(f"It moved up about {change_pct:.2f}% vs previous close.")
    else:
        parts.append(f"It moved down about {abs(change_pct):.2f}% vs previous close.")
    parts.append(f"Price is {range_comment}.")
    parts.append(f"Overall risk profile looks {risk_level.lower()} based on recent volatility.")
    if live_hint_text:
        parts.append(live_hint_text)
    parts.append(
        "Always pair this AI-style view with your own research, risk appetite, and time horizon."
    )

    opinion = " ".join(parts)

    return {
        "symbol": user_sym,
        "data_symbol_used": yahoo_sym,
        "last_close": round(last_close, 2),
        "change_percent": round(change_pct, 2),
        "short_term_view": short_view,
        "long_term_view": long_view,
        "risk_level": risk_level,
        "hi_52w": round(hi_52w, 2),
        "lo_52w": round(lo_52w, 2),
        "ai_opinion": opinion,
        "live_hint": live_hint_text,  # may be null if Kite not configured
        "disclaimer": (
            "Prices are derived using yfinance (Yahoo Finance data) and may be delayed/inaccurate. "
            "Intraday risk hints (if present) may use your private broker data on the server side. "
            "Fineryx AI is for informational purposes only, not investment advice."
        ),
    }
