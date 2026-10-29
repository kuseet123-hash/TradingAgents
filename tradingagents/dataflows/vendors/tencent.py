"""Explicit Tencent raw daily feed for SH/SZ mainboard and ChiNext equities.

The protocol is verified against AKShare's stock_hist_tx.py at 68f2977203a0.
Raw volume is contracted as lots and raw amount as CNY 10,000 yuan. In
particular sz000 equities are NOT exempted from the lots conversion. Every
used bar must pass an amount/volume/range check; factors are never inferred.
STAR-board units are not verified here and are deliberately unavailable.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from tradingagents.dataflows.ashare_market import (
    SHANGHAI,
    anonymous_auth,
    completed_window,
    finish_bars,
    number,
    parse_date,
    price_header,
    render_indicators,
    render_snapshot,
)
from tradingagents.dataflows.errors import VendorUnavailableError

PRICE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
MAX_PRICE_DAYS = 730
MAX_ROWS = 640


def _now():
    return datetime.now(SHANGHAI)


def _get_payload(params: dict) -> dict:
    with requests.Session() as session:
        session.auth = anonymous_auth
        session.mount("https://", HTTPAdapter(max_retries=0))
        try:
            response = session.get(
                PRICE_URL,
                params=params,
                timeout=(5, 8),
                verify=True,
                allow_redirects=False,
                headers={"User-Agent": "TradingAgents-Ashare/1.0 (read-only research)"},
            )
            if response.status_code != 200:
                raise VendorUnavailableError(
                    f"Tencent HTTP {response.status_code}; no retry or access bypass"
                )
            text = response.text.strip()
            prefix = params["_var"] + "="
            if text.startswith(prefix):
                text = text[len(prefix) :]
            payload = json.loads(text.rstrip(";"))
        except (requests.RequestException, ValueError) as exc:
            raise VendorUnavailableError(
                f"Tencent request/JSON unavailable ({type(exc).__name__})"
            ) from exc
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise VendorUnavailableError("Tencent response did not succeed")
    return payload


def _unit_check(low, high, shares, yuan):
    """Check fixed units using independent source fields, never generate amount."""
    if shares == 0 and yuan == 0:
        return False  # this row cannot establish units, but need not contradict them
    if shares <= 0 or yuan <= 0:
        raise ValueError("inconsistent zero volume/amount")
    vwap = yuan / shares
    # Cap absolute rounding tolerance by 1% of low, including penny-priced rows.
    tolerance = min(0.02, low * 0.01)
    if not low - tolerance <= vwap <= high + tolerance:
        raise ValueError("fixed lots/yuan unit contract fails amount/volume/price check")
    return True


def load_daily_bars(symbol, start_date, end_date, *, adjustment="none") -> pd.DataFrame:
    stock, start, final_end, sessions = completed_window(
        symbol, start_date, end_date, adjustment, _now(), max_days=MAX_PRICE_DAYS
    )
    if stock.code.startswith("688"):
        raise VendorUnavailableError("Tencent STAR-board volume units unverified; data withheld")
    if len(sessions) > MAX_ROWS:
        raise ValueError(f"Tencent window exceeds {MAX_ROWS} trading rows")
    tx_symbol = stock.exchange.lower() + stock.code
    payload = _get_payload(
        {
            "_var": "kline_day" + str(final_end.year),
            "param": f"{tx_symbol},day,{start},{final_end},{min(MAX_ROWS, max(20, len(sessions) + 5))},",
            "r": "0.8205512681390605",
        }
    )
    data = payload.get("data")
    entry = data.get(tx_symbol) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        raise VendorUnavailableError("Tencent response has no matching instrument")
    daily = entry.get("day")
    # Never substitute qfqday/hfqday when a raw series is missing.
    if not isinstance(daily, list) or len(daily) > MAX_ROWS:
        raise VendorUnavailableError("Tencent unadjusted day schema/coverage unavailable")
    rows, seen = [], set()
    unit_checks = 0
    try:
        for fields in daily:
            if not isinstance(fields, list) or len(fields) != 11:
                raise ValueError("expected 11 daily fields including independent amount")
            day = parse_date(fields[0])
            if not start <= day <= final_end:
                continue
            if day not in sessions or day in seen:
                raise ValueError("non-session or duplicate daily row")
            seen.add(day)
            open_, close, high, low, lots = map(number, fields[1:6])
            shares = number(Decimal(str(fields[5])) * 100)
            turnover = number(Decimal(str(fields[7])) / 100)
            amount_raw = number(fields[8])
            yuan = number(Decimal(str(fields[8])) * 10000)
            if not 0 < low <= min(open_, close) <= max(open_, close) <= high:
                raise ValueError("invalid OHLC range")
            unit_checks += _unit_check(low, high, shares, yuan)
            rows.append(
                [
                    pd.Timestamp(day),
                    open_,
                    high,
                    low,
                    close,
                    shares,
                    yuan,
                    lots,
                    amount_raw,
                    turnover,
                ]
            )
    except (ValueError, TypeError, AttributeError, InvalidOperation) as exc:
        raise VendorUnavailableError(f"Tencent daily fields/units unverified: {exc}") from exc
    if rows and not unit_checks:
        raise VendorUnavailableError("Tencent units cannot be verified from zero-trade rows")
    return finish_bars(
        symbol,
        rows,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Amount",
            "VolumeLots",
            "AmountTenThousandYuan",
            "Turnover",
        ],
        sessions,
        source="Tencent public unadjusted daily feed",
        retrieved_at=_now().isoformat(),
        source_units="volume lots (including sz000); amount 10000 yuan; turnover percent",
        unit_check="fixed factors checked against source amount/volume and daily low/high",
        verified_unit_rows=unit_checks,
        response_rows=len(daily),
    )


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    frame = load_daily_bars(symbol, start_date, end_date)
    return price_header(symbol, frame) + frame.to_csv(index=False)


def get_indicators(symbol: str, indicator: str, as_of_date: str, look_back_days: int) -> str:
    return render_indicators(symbol, indicator, as_of_date, look_back_days, load_daily_bars)


def build_verified_market_snapshot(symbol: str, as_of_date: str, look_back_days: int = 30) -> str:
    return render_snapshot(symbol, as_of_date, look_back_days, load_daily_bars)
