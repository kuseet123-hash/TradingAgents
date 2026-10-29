"""Shared completed-session and indicator rules for domestic price vendors."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
from exchange_calendars.exchange_calendar_xshg import XSHGExchangeCalendar
from stockstats import wrap

from tradingagents.dataflows.ashares import parse_ashare_symbol
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError

SHANGHAI = ZoneInfo("Asia/Shanghai")
INDICATORS = {
    "close_10_ema": 10,
    "close_50_sma": 50,
    "close_200_sma": 200,
    "macd": 35,
    "macds": 35,
    "macdh": 35,
    "rsi": 15,
    "boll": 20,
    "boll_ub": 20,
    "boll_lb": 20,
    "atr": 15,
    "vwma": 10,
    "mfi": 15,
}


def anonymous_auth(request):
    """Public feeds must not look up per-host credentials in the user's netrc."""
    return request


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


@lru_cache(maxsize=1)
def _calendar():
    return XSHGExchangeCalendar(
        start=XSHGExchangeCalendar.bound_min(), end=XSHGExchangeCalendar.bound_max()
    )


def sessions_between(start, end) -> set:
    calendar = _calendar()
    if start < calendar.first_session.date() or end > calendar.last_session.date():
        raise VendorUnavailableError(
            f"A-share calendar unavailable outside {calendar.first_session.date()}.."
            f"{calendar.last_session.date()}; update exchange-calendars"
        )
    return set(calendar.sessions_in_range(str(start), str(end)).date)


def completed_window(symbol, start_date, end_date, adjustment, now, *, max_days):
    stock = parse_ashare_symbol(symbol)
    if adjustment != "none":
        raise ValueError("Only adjustment='none' is supported; qfq/hfq have no as-of vintage")
    start, end = parse_date(start_date), parse_date(end_date)
    if start > end:
        raise ValueError("start_date is after end_date")
    if end > now.date():
        raise ValueError("A-share end_date must not be in the future (Asia/Shanghai)")
    if (end - start).days > max_days:
        raise ValueError(f"A-share price window exceeds {max_days} days")
    final_end = min(end, now.date() - timedelta(days=1))
    sessions = sessions_between(start, final_end) if start <= final_end else set()
    if not sessions:
        raise NoMarketDataError(symbol, stock.canonical, "no completed sessions in window")
    return stock, start, final_end, sessions


def number(value) -> float:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("non-finite or negative market value")
    converted = float(result)
    if not pd.notna(converted) or converted == float("inf"):
        raise ValueError("market value out of range")
    return converted


def finish_bars(symbol, rows, columns, sessions, *, source, retrieved_at, **metadata):
    stock = parse_ashare_symbol(symbol)
    if not rows:
        raise NoMarketDataError(symbol, stock.canonical, "no usable completed daily rows")
    frame = pd.DataFrame(rows, columns=columns).sort_values("Date").reset_index(drop=True)
    if frame.iloc[-1]["Date"].date() != max(sessions):
        raise NoMarketDataError(
            symbol,
            stock.canonical,
            f"latest row {frame.iloc[-1]['Date'].date()} is before expected session "
            f"{max(sessions)}; suspension/coverage unknown, no forward fill",
        )
    frame.attrs.update(
        source=source,
        adjustment="none",
        timezone="Asia/Shanghai",
        retrieved_at=retrieved_at,
        **metadata,
    )
    return frame


def price_header(symbol: str, frame: pd.DataFrame) -> str:
    return (
        f"# A-share {parse_ashare_symbol(symbol).canonical}; source: {frame.attrs['source']}\n"
        f"# Retrieved: {frame.attrs['retrieved_at']}; timezone: Asia/Shanghai; adjustment: none\n"
        "# OHLC: CNY yuan/share; Volume: shares = VolumeLots * 100; Amount: CNY yuan (not 万元)\n"
        f"# Source units: {frame.attrs.get('source_units', 'volume lots; amount yuan')}\n"
        "# Prior completed sessions only; today's provisional row is withheld even after close.\n"
        "# Missing rows are not filled; source corrections may change history; no archived vintage.\n"
        "# Unadjusted price gaps from corporate actions are not investment returns.\n"
    )


def _indicator_frame(symbol, as_of_date, loader):
    end = parse_date(as_of_date)
    return loader(symbol, str(end - timedelta(days=730)), str(end))


def _indicator_value(stock_frame, indicator, position):
    if position + 1 < INDICATORS[indicator]:
        return "N/A: insufficient trading rows for warmup"
    value = stock_frame[indicator].iloc[position]
    return f"{value:.8g}" if pd.notna(value) and abs(value) != float("inf") else "N/A"


def render_indicators(symbol, indicator, as_of_date, look_back_days, loader):
    if indicator not in INDICATORS:
        raise ValueError(f"Unsupported A-share indicator: {indicator}")
    if not isinstance(look_back_days, int) or not 0 <= look_back_days <= 365:
        raise ValueError("A-share look_back_days must be in 0..365")
    frame = _indicator_frame(symbol, as_of_date, loader)
    stock_frame = wrap(frame.copy())
    start = pd.Timestamp(parse_date(as_of_date) - timedelta(days=look_back_days))
    lines = [price_header(symbol, frame), f"# Indicator: {indicator}; only observed trading rows"]
    for position, row in frame.iterrows():
        if row["Date"] >= start:
            lines.append(
                f"{row['Date']:%Y-%m-%d}: {_indicator_value(stock_frame, indicator, position)}"
            )
    return "\n".join(lines)


def render_snapshot(symbol, as_of_date, look_back_days, loader):
    frame = _indicator_frame(symbol, as_of_date, loader)
    stock_frame = wrap(frame.copy())
    lines = [
        price_header(symbol, frame),
        f"## Verified market data snapshot as of {as_of_date}",
        f"Latest trading row used: {frame.iloc[-1]['Date']:%Y-%m-%d}",
        frame.tail(max(1, min(int(look_back_days), 30))).to_csv(index=False),
        "| Indicator | Value |",
        "|---|---:|",
    ]
    for indicator in INDICATORS:
        lines.append(
            f"| {indicator} | {_indicator_value(stock_frame, indicator, len(frame) - 1)} |"
        )
    return "\n".join(lines)
