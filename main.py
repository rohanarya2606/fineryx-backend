import os
import math
from statistics import mean
from typing import List, Optional

import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional: KiteConnect for extra intraday signals (for YOU only)
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

app = FastAPI()

# CORS – open for now (can tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#  KEEPALIVE: PING ENDPOINT (NEW)
# =========================

@app.get("/ping")
def ping():
    """
    Lightweight health endpoint for UptimeRobot / keepalive.
    """
    return {"status": "ok", "message": "Fineryx backend warm 🔥"}


# ========= OPTIONAL AUTO-WARMUP ON STARTUP (NEW) =========

@app.on_event("startup")
def warmup():
    """
    This runs ONCE when Render starts the container.
    It does a tiny yfinance call to warm imports, DNS, SSL, etc.
    Helps reduce cold-start delay on the first real user request.
    """
    try:
        print("🔥 Fineryx Warmup: starting warmup fetch...")
        ticker = yf.Ticker("RELIANCE.NS")  # any common, liquid symbol
        ticker.history(period="1d", interval="1d")  # very light
        print("🔥 Fineryx Warmup: completed successfully.")
    except Exception as e:
        print("⚠️ Warmup error:", e)


# =========================
#  HELPER: SYMBOL RESOLUTION
# =========================

def resolve_symbol(user_symbol: str):
    """
    Map user input to a working yfinance symbol.

    Strategy:
      - If user includes '.', treat it as full ticker (e.g. RELIANCE.NS, AAPL)
      - Else try: SYM.NS (NSE), then SYM.BO (BSE), then SYM (no suffix)
    Returns:
      (clean_user_symbol, yahoo_symbol, error_message_or_None)
    """
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
    """
    Return a yfinance history DataFrame or dict(error=...).
    """
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
            "/health",
        ],
    }

# =========================
#  1) STOCK ANALYZER
# =========================

@app.get("/api/analyze-stock")
def analyze_stock(symbol: str):
    """
    Stock analyzer: trend, risk, 52-week, AI-style summary.
    """
    user_sym, yahoo_sym, sym_error = resolve_symbol(symbol)

    if sym_error:
        return {
            "error": sym_error,
            "symbol": user_sym or symbol,
        }

    df = fetch_history_df(yahoo_sym, period="1y", interval="1d")
    if isinstance(df, dict) and "error" in df:
        return {
            "error": df["error"],
            "symbol": user_sym,
            "data_symbol_used": yahoo_sym,
        }

    closes = closes_from_df(df)
    if isinstance(closes, dict) and "error" in closes:
        return {
            "error": closes["error"],
            "symbol": user_sym,
            "data_symbol_used": yahoo_sym,
        }

    # Basic prices
    last_close = closes[-1]
    prev_close = closes[-2]
    change_pct = 0.0 if prev_close == 0 else (last_close - prev_close) / prev_close * 100

    # SMAs
    def sma(arr, n: int):
        if len(arr) < n:
            return None
        return mean(arr[-n:])

    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)

    # Short-term trend
    if sma20 and sma50:
        if sma20 > sma50 * 1.01:
            short_view = "Bullish"
        elif sma20 < sma50 * 0.99:
            short_view = "Bearish"
        else:
            short_view = "Sideways"
    else:
        short_view = "Not enough data"

    # Long-term trend
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

    # Historical vol
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

    # Optional intraday refinement via Kite
    kite_hint = get_kite_intraday_risk(user_sym)
    live_hint_text = None
    if kite_hint and "level" in kite_hint:
        kite_level = kite_hint["level"]
        order = {"Low": 0, "Medium": 1, "High": 2}
        if kite_level in order and risk_level in order:
            if order[kite_level] > order[risk_level]:
                risk_level = kite_level
        live_hint_text = f"Intraday broker feed suggests {kite_level.lower()} volatility today."

    # Opinion
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
        "live_hint": live_hint_text,
        "disclaimer": (
            "Prices are derived using yfinance (Yahoo Finance data) and may be delayed/inaccurate. "
            "Intraday risk hints (if present) may use your private broker data on the server side. "
            "Fineryx AI is for informational purposes only, not investment advice."
        ),
    }

# =========================
#  2) MARKET SENTIMENT
# =========================

@app.get("/api/market-sentiment")
def market_sentiment():
    """
    Fineryx Market Sentiment:
    - Uses Nifty, BankNifty, and India VIX.
    """
    indices = {
        "nifty": "^NSEI",
        "banknifty": "^NSEBANK",
        "vix": "INDIAVIX.NS",
    }

    out = {}
    for key, ticker in indices.items():
        df = fetch_history_df(ticker, period="5d", interval="1d")
        if isinstance(df, dict) and "error" in df:
            out[key] = {"error": df["error"]}
        else:
            out[key] = df

    # Nifty change
    nifty_change = None
    if not isinstance(out["nifty"], dict):
        closes = out["nifty"]["Close"].dropna().tolist()
        if len(closes) >= 2:
            nifty_change = (closes[-1] - closes[-2]) / closes[-2] * 100

    # BankNifty change
    banknifty_change = None
    if not isinstance(out["banknifty"], dict):
        closes = out["banknifty"]["Close"].dropna().tolist()
        if len(closes) >= 2:
            banknifty_change = (closes[-1] - closes[-2]) / closes[-2] * 100

    # VIX last value
    vix_value = None
    if not isinstance(out["vix"], dict):
        closes = out["vix"]["Close"].dropna().tolist()
        if len(closes) >= 1:
            vix_value = closes[-1]

    # VIX mood
    if vix_value is None:
        vix_bucket = "Unknown"
    elif vix_value < 12:
        vix_bucket = "Calm"
    elif vix_value < 16:
        vix_bucket = "Normal"
    elif vix_value < 22:
        vix_bucket = "Elevated"
    else:
        vix_bucket = "High Volatility"

    # Nifty-based overall mood
    if nifty_change is None:
        mood = "Unknown"
    elif nifty_change > 0.75:
        mood = "Bullish"
    elif nifty_change > 0.25:
        mood = "Mild Bullish"
    elif nifty_change > -0.25:
        mood = "Neutral"
    elif nifty_change > -0.75:
        mood = "Mild Bearish"
    else:
        mood = "Bearish"

    # Confidence via BankNifty
    if banknifty_change is None or nifty_change is None:
        confidence = "Unknown"
    else:
        same_side = (nifty_change >= 0 and banknifty_change >= 0) or (
            nifty_change <= 0 and banknifty_change <= 0
        )
        confidence = "Strong" if same_side else "Mixed"

    summary_parts = []
    if nifty_change is not None:
        summary_parts.append(
            f"Market mood looks {mood.lower()} with Nifty move around {nifty_change:.2f}%."
        )
    else:
        summary_parts.append("Market mood is unclear.")
    if banknifty_change is not None:
        summary_parts.append(f"BankNifty is at {banknifty_change:.2f}% move.")
    if vix_value is not None:
        summary_parts.append(
            f"India VIX is {vix_value:.2f}, indicating {vix_bucket.lower()} volatility."
        )
    summary = " ".join(summary_parts)

    return {
        "mood": mood,
        "confidence": confidence,
        "nifty_change": round(nifty_change, 2) if nifty_change is not None else None,
        "banknifty_change": round(banknifty_change, 2) if banknifty_change is not None else None,
        "vix": round(vix_value, 2) if vix_value is not None else None,
        "vix_bucket": vix_bucket,
        "summary": summary,
    }

# =========================
#  3) AI STOCK PICKS (MOMENTUM)
# =========================

# Simple initial universe – expand later if you want
NIFTY_CANDIDATES = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "AXISBANK", "LT", "ITC", "KOTAKBANK",
    "ASIANPAINT", "HINDUNILVR", "BAJFINANCE", "ULTRACEMCO",
    "MARUTI", "SUNPHARMA", "TITAN", "POWERGRID", "NTPC"
]

@app.get("/api/ai-picks")
def ai_picks(limit: int = 5):
    """
    Fineryx AI Stock Picks: momentum-based scanner over a small universe.
    """
    picks = []

    for sym in NIFTY_CANDIDATES:
        user_sym, yahoo_sym, sym_error = resolve_symbol(sym)
        if sym_error or not yahoo_sym:
            continue

        df = fetch_history_df(yahoo_sym, period="1y", interval="1d")
        if isinstance(df, dict) and "error" in df:
            continue

        closes = df["Close"].dropna().tolist()
        if len(closes) < 60:
            continue

        last_close = closes[-1]
        hi_52w = max(closes)

        # basic SMAs
        def sma(arr, n):
            return mean(arr[-n:]) if len(arr) >= n else None

        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)

        if sma20 is None or sma50 is None:
            continue

        momentum_strength = (sma20 - sma50) / sma50 * 100

        # 52w high proximity
        dist_high = (hi_52w - last_close) / hi_52w * 100 if hi_52w else 999

        # daily vol
        returns = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            cur = closes[i]
            if prev:
                returns.append((cur - prev) / prev * 100)
        if not returns:
            continue

        avg_ret = sum(returns) / len(returns)
        var = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        vol = math.sqrt(var)

        # Reject too volatile
        if vol > 3.5:
            continue

        # Trend score (simple)
        trend_score = 2 if sma20 > sma50 else 0

        volatility_penalty = max(0, vol - 1.5)

        score = (momentum_strength * 2) + trend_score - volatility_penalty

        picks.append({
            "symbol": user_sym,
            "data_symbol_used": yahoo_sym,
            "last_close": round(last_close, 2),
            "momentum_strength": round(momentum_strength, 2),
            "distance_from_high_pct": round(dist_high, 2),
            "volatility": round(vol, 2),
            "score": round(score, 2),
        })

    # Sort & limit
    picks_sorted = sorted(picks, key=lambda x: x["score"], reverse=True)
    picks_out = picks_sorted[: max(1, min(limit, 10))]

    summary = (
        "These stocks appear to show relatively clean price momentum with controlled volatility "
        "based on Fineryx's internal momentum model."
    )

    return {
        "universe_size": len(NIFTY_CANDIDATES),
        "picked": len(picks_out),
        "summary": summary,
        "picks": picks_out,
    }

# =========================
#  4) VOLATILITY CHECKER
# =========================

@app.get("/api/volatility-checker")
def volatility_checker(symbol: str):
    """
    Classify stock as Very Calm / Calm / Moderate / Volatile / Highly Volatile
    and provide ATR-based confirmation.
    """
    user_sym, yahoo_sym, sym_error = resolve_symbol(symbol)
    if sym_error:
        return {"error": sym_error, "symbol": user_sym or symbol}

    df = fetch_history_df(yahoo_sym, period="1y", interval="1d")
    if isinstance(df, dict) and "error" in df:
        return {"error": df["error"], "symbol": user_sym, "data_symbol_used": yahoo_sym}

    closes = df["Close"].dropna()
    if len(closes) < 30:
        return {"error": "Not enough history for volatility analysis.", "symbol": user_sym}

    prices = closes.tolist()
    # daily returns
    rets = []
    for i in range(1, len(prices)):
        if prices[i - 1]:
            rets.append((prices[i] - prices[i - 1]) / prices[i - 1] * 100)

    if not rets:
        return {"error": "Unable to compute returns.", "symbol": user_sym}

    avg_ret = sum(rets) / len(rets)
    var = sum((r - avg_ret) ** 2 for r in rets) / len(rets)
    vol = math.sqrt(var)

    # Vol buckets
    if vol < 1.2:
        vol_bucket = "Very Calm"
    elif vol < 2.2:
        vol_bucket = "Calm"
    elif vol < 3.5:
        vol_bucket = "Moderate"
    elif vol < 5:
        vol_bucket = "Volatile"
    else:
        vol_bucket = "Highly Volatile"

    # ATR 14
    try:
        highs = df["High"].tolist()
        lows = df["Low"].tolist()
        closes_list = df["Close"].tolist()
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes_list[i - 1]),
                abs(lows[i] - closes_list[i - 1]),
            )
            trs.append(tr)
        if len(trs) >= 14:
            atr14 = sum(trs[-14:]) / 14.0
        else:
            atr14 = sum(trs) / len(trs) if trs else 0.0
    except Exception:
        atr14 = 0.0

    last_close = prices[-1]
    atr_pct = (atr14 / last_close * 100) if last_close else 0.0

    # ATR bucket
    if atr_pct < 1.5:
        atr_bucket = "Stable"
    elif atr_pct < 3:
        atr_bucket = "Moderate"
    else:
        atr_bucket = "Wild"

    summary = (
        f"{user_sym} shows {vol_bucket.lower()} historical volatility (~{vol:.2f}% daily) "
        f"with ATR around {atr_pct:.2f}% of price, indicating {atr_bucket.lower()} intraday structure."
    )

    return {
        "symbol": user_sym,
        "data_symbol_used": yahoo_sym,
        "historical_volatility_pct": round(vol, 2),
        "volatility_bucket": vol_bucket,
        "atr_14": round(atr14, 2),
        "atr_pct_of_price": round(atr_pct, 2),
        "atr_bucket": atr_bucket,
        "summary": summary,
    }

# =========================
#  5) PORTFOLIO RISK CHECKER
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


class Holding(BaseModel):
    symbol: str
    weight: Optional[float] = None  # percentage like 25.0


class PortfolioRequest(BaseModel):
    holdings: List[Holding]


@app.post("/api/portfolio-risk")
def portfolio_risk(req: PortfolioRequest):
    """
    Simple portfolio risk checker:
    - Uses provided weights (percentages).
    - Computes volatility-weighted risk.
    - Flags concentration.
    """
    if not req.holdings:
        return {"error": "Provide at least one holding."}

    # Normalize weights if missing or not summing to 100
    weights = []
    symbols = []
    for h in req.holdings:
        if not h.symbol:
            continue
        symbols.append(h.symbol.upper())
        weights.append(h.weight if h.weight is not None else 0.0)

    if not symbols:
        return {"error": "No valid symbols provided."}

    total_w = sum(weights)
    if total_w <= 0:
        # equal weights
        weights = [1.0 / len(symbols)] * len(symbols)
    else:
        weights = [w / total_w for w in weights]

    # Compute per-stock vol
    stock_infos = []
    for sym, w in zip(symbols, weights):
        user_sym, yahoo_sym, sym_error = resolve_symbol(sym)
        if sym_error or not yahoo_sym:
            stock_infos.append(
                {"symbol": user_sym or sym, "weight": round(w * 100, 2), "error": sym_error or "Resolution failed"}
            )
            continue

        df = fetch_history_df(yahoo_sym, period="1y", interval="1d")
        if isinstance(df, dict) and "error" in df:
            stock_infos.append(
                {"symbol": user_sym, "weight": round(w * 100, 2), "error": df["error"]}
            )
            continue

        closes = df["Close"].dropna().tolist()
        if len(closes) < 30:
            stock_infos.append(
                {"symbol": user_sym, "weight": round(w * 100, 2), "error": "Not enough data"}
            )
            continue

        rets = []
        for i in range(1, len(closes)):
            if closes[i - 1]:
                rets.append((closes[i] - closes[i - 1]) / closes[i - 1] * 100)
        if not rets:
            stock_infos.append(
                {"symbol": user_sym, "weight": round(w * 100, 2), "error": "Unable to compute returns"}
            )
            continue

        avg_ret = sum(rets) / len(rets)
        var = sum((r - avg_ret) ** 2 for r in rets) / len(rets)
        vol = math.sqrt(var)

        stock_infos.append(
            {
                "symbol": user_sym,
                "data_symbol_used": yahoo_sym,
                "weight": round(w * 100, 2),
                "volatility_pct": round(vol, 2),
            }
        )

    # portfolio risk = sum(weight * vol)
    portfolio_risk_val = 0.0
    for info in stock_infos:
        if "volatility_pct" in info and "weight" in info:
            portfolio_risk_val += (info["weight"] / 100.0) * info["volatility_pct"]

    # risk buckets
    if portfolio_risk_val < 1.8:
        risk_bucket = "Low"
    elif portfolio_risk_val < 3:
        risk_bucket = "Medium"
    else:
        risk_bucket = "High"

    # concentration flag
    warnings = []
    for info in stock_infos:
        if "weight" in info and info["weight"] > 35:
            warnings.append(f"{info['symbol']} is {info['weight']}% of portfolio (high concentration).")

    summary_parts = [
        f"Portfolio risk looks {risk_bucket.lower()} with volatility-weighted score around {portfolio_risk_val:.2f}."
    ]
    if warnings:
        summary_parts.append(" ".join(warnings))

    summary = " ".join(summary_parts)

    return {
        "portfolio_risk_score": round(portfolio_risk_val, 2),
        "portfolio_risk_bucket": risk_bucket,
        "holdings": stock_infos,
        "warnings": warnings,
        "summary": summary,
        "disclaimer": (
            "Portfolio analysis is based on historical price volatility only and does not account for all risks. "
            "Informational use only, not investment advice."
        ),
    }
