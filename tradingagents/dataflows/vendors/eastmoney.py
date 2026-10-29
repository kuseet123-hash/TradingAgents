"""Bounded, credential-free SH/SZ research reads from Eastmoney public feeds.

Only unadjusted, completed prior-session prices and a disclosure index are
served. The index is not a versioned financial database or a news archive.
Requests stop on access restrictions; no cookies, retry loop or auth bypass.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from exchange_calendars.exchange_calendar_xshg import XSHGExchangeCalendar
from stockstats import wrap

from tradingagents.dataflows.ashares import parse_ashare_symbol
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError

SHANGHAI = ZoneInfo("Asia/Shanghai")
PRICE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
ANNOUNCEMENT_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
MAX_PRICE_DAYS = 3660
MAX_ANNOUNCEMENT_PAGES = 3
PAGE_SIZE = 100
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


def _now() -> datetime:
    return datetime.now(SHANGHAI)


def _date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _window(start_date: str, end_date: str):
    start, end = _date(start_date), _date(end_date)
    if start > end:
        raise ValueError("start_date is after end_date")
    if end > _now().date():
        raise ValueError("A-share end_date must not be in the future (Asia/Shanghai)")
    return start, end


def _get_json(url: str, params: dict) -> dict:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
            headers={"User-Agent": "TradingAgents-Ashare/1.0 (read-only research)"},
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise VendorUnavailableError(
                f"Eastmoney HTTP {response.status_code}; no retry or access bypass"
            )
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise VendorUnavailableError("Eastmoney request/JSON unavailable") from exc
    if not isinstance(payload, dict):
        raise VendorUnavailableError("Eastmoney response schema changed")
    return payload


@lru_cache(maxsize=1)
def _calendar():
    # Finite published holiday coverage: never guess weekdays beyond its bound.
    return XSHGExchangeCalendar(
        start=XSHGExchangeCalendar.bound_min(), end=XSHGExchangeCalendar.bound_max()
    )


def _sessions(start, end) -> set:
    calendar = _calendar()
    if start < calendar.first_session.date() or end > calendar.last_session.date():
        raise VendorUnavailableError(
            f"A-share calendar unavailable outside {calendar.first_session.date()}.."
            f"{calendar.last_session.date()}; update exchange-calendars"
        )
    return set(calendar.sessions_in_range(str(start), str(end)).date)


def _number(value: str) -> float:
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("non-finite or negative market value")
    converted = float(result)
    if not pd.notna(converted) or converted == float("inf"):
        raise ValueError("market value out of range")
    return converted


def load_daily_bars(
    symbol: str, start_date: str, end_date: str, *, adjustment: str = "none"
) -> pd.DataFrame:
    """Inclusive window. Volume=shares (source lots*100), Amount=CNY yuan.

    Today's row is always withheld, even after 15:00: this feed has no final-bar
    flag. A missing last expected session is unavailable (possibly suspended),
    never filled with an earlier close. SH and SZ share the regular calendar.
    """
    stock = parse_ashare_symbol(symbol)
    if adjustment != "none":
        raise ValueError("Only adjustment='none' is supported; qfq/hfq have no as-of vintage")
    start, end = _window(start_date, end_date)
    if (end - start).days > MAX_PRICE_DAYS:
        raise ValueError(f"A-share price window exceeds {MAX_PRICE_DAYS} days")
    final_end = min(end, _now().date() - timedelta(days=1))
    sessions = _sessions(start, final_end) if start <= final_end else set()
    if not sessions:
        raise NoMarketDataError(symbol, stock.canonical, "no completed sessions in window")
    payload = _get_json(
        PRICE_URL,
        {
            "secid": stock.secid,
            "klt": "101",
            "fqt": "0",
            "beg": start.strftime("%Y%m%d"),
            "end": final_end.strftime("%Y%m%d"),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        },
    )
    if payload.get("rc") != 0:
        raise VendorUnavailableError("Eastmoney price response did not succeed")
    data = payload.get("data")
    if data is None:
        raise NoMarketDataError(symbol, stock.canonical, "source returned no price data")
    if (
        not isinstance(data, dict)
        or data.get("code") != stock.code
        or data.get("market") != int(stock.secid[0])
    ):
        raise VendorUnavailableError("Eastmoney returned a mismatched instrument")
    klines = data.get("klines")
    if not isinstance(klines, list):
        raise VendorUnavailableError("Eastmoney price schema changed")
    rows = []
    seen = set()
    try:
        for line in klines:
            fields = line.split(",")
            if len(fields) != 11:
                raise ValueError("unexpected daily fields")
            day = _date(fields[0])
            if not start <= day <= final_end:
                continue  # defensive cutoff even if vendor ignores the window
            if day not in sessions or day in seen:
                raise ValueError("non-session or duplicate daily row")
            seen.add(day)
            open_, close, high, low, lots, amount = map(_number, fields[1:7])
            shares = _number(str(Decimal(fields[5]) * 100))
            if not 0 < low <= min(open_, close) <= max(open_, close) <= high:
                raise ValueError("invalid OHLC range")
            rows.append([pd.Timestamp(day), open_, high, low, close, shares, amount, lots])
    except (ValueError, TypeError, AttributeError, InvalidOperation) as exc:
        raise VendorUnavailableError("Eastmoney malformed daily row; values withheld") from exc
    if not rows:
        raise NoMarketDataError(symbol, stock.canonical, "no usable completed daily rows")
    frame = (
        pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Amount", "VolumeLots"]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if frame.iloc[-1]["Date"].date() != max(sessions):
        raise NoMarketDataError(
            symbol,
            stock.canonical,
            f"latest row {frame.iloc[-1]['Date'].date()} is before expected session "
            f"{max(sessions)}; suspension/coverage unknown, no forward fill",
        )
    frame.attrs.update(
        source="Eastmoney public daily feed",
        adjustment="none",
        timezone="Asia/Shanghai",
        retrieved_at=_now().isoformat(),
    )
    return frame


def _price_header(symbol: str, frame: pd.DataFrame) -> str:
    return (
        f"# A-share {parse_ashare_symbol(symbol).canonical}; source: Eastmoney public daily feed\n"
        f"# Retrieved: {frame.attrs['retrieved_at']}; timezone: Asia/Shanghai; adjustment: none\n"
        "# OHLC: CNY yuan/share; Volume: shares = VolumeLots * 100; Amount: CNY yuan (not 万元)\n"
        "# Prior completed sessions only; today's provisional row is withheld even after close.\n"
        "# Missing rows are not filled; source corrections may change history; no archived vintage.\n"
        "# Unadjusted price gaps from corporate actions are not investment returns.\n"
    )


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    frame = load_daily_bars(symbol, start_date, end_date)
    return _price_header(symbol, frame) + frame.to_csv(index=False)


def _indicator_frame(symbol: str, as_of_date: str):
    end = _date(as_of_date)
    return load_daily_bars(symbol, str(end - timedelta(days=730)), str(end))


def _indicator_value(frame, indicator: str, position: int) -> str:
    if indicator not in INDICATORS:
        raise ValueError(f"Unsupported A-share indicator: {indicator}")
    if position + 1 < INDICATORS[indicator]:
        return "N/A: insufficient trading rows for warmup"
    series = wrap(frame.copy())[indicator]
    value = series.iloc[position]
    return f"{value:.8g}" if pd.notna(value) and abs(value) != float("inf") else "N/A"


def get_indicators(symbol: str, indicator: str, as_of_date: str, look_back_days: int) -> str:
    if not isinstance(look_back_days, int) or not 0 <= look_back_days <= 365:
        raise ValueError("A-share look_back_days must be in 0..365")
    frame = _indicator_frame(symbol, as_of_date)
    start = pd.Timestamp(_date(as_of_date) - timedelta(days=look_back_days))
    lines = [_price_header(symbol, frame), f"# Indicator: {indicator}; only observed trading rows"]
    for position, row in frame.iterrows():
        if row["Date"] >= start:
            lines.append(f"{row['Date']:%Y-%m-%d}: {_indicator_value(frame, indicator, position)}")
    return "\n".join(lines)


def build_verified_market_snapshot(symbol: str, as_of_date: str, look_back_days: int = 30) -> str:
    frame = _indicator_frame(symbol, as_of_date)
    latest = frame.iloc[-1]
    lines = [
        _price_header(symbol, frame),
        f"## Verified market data snapshot as of {as_of_date}",
        f"Latest trading row used: {latest['Date']:%Y-%m-%d}",
        frame.tail(max(1, min(int(look_back_days), 30))).to_csv(index=False),
        "| Indicator | Value |",
        "|---|---:|",
    ]
    for indicator in INDICATORS:
        lines.append(f"| {indicator} | {_indicator_value(frame, indicator, len(frame) - 1)} |")
    return "\n".join(lines)


def _publication_time(value):
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S:%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=SHANGHAI)
        except ValueError:
            continue
    return None


def _report_period(title: str) -> str | None:
    match = re.search(r"(\d{4})年(年度|半年度|第一季度|第三季度)报告", title)
    if not match:
        return None
    return (
        match[1]
        + {"年度": "-12-31", "半年度": "-06-30", "第一季度": "-03-31", "第三季度": "-09-30"}[
            match[2]
        ]
    )


def get_announcements(
    symbol: str, start_date: str, end_date: str, *, reports_only=False, freq: str | None = None
) -> str:
    """Current index filtered by disclosure availability, not an archived snapshot."""
    stock = parse_ashare_symbol(symbol)
    start, end = _window(start_date, end_date)
    if (end - start).days > 730:
        raise ValueError("A-share announcement window exceeds 730 days")
    if freq not in (None, "annual", "quarterly"):
        raise ValueError("freq must be annual or quarterly")
    cutoff = min(datetime.combine(end, time.max, SHANGHAI), _now())
    rows, seen = [], set()
    excluded_unknown = 0
    truncated = False
    for page in range(1, MAX_ANNOUNCEMENT_PAGES + 1):
        payload = _get_json(
            ANNOUNCEMENT_URL,
            {
                "stock_list": stock.code,
                "ann_type": "A",
                "sr": "-1",
                "page_size": PAGE_SIZE,
                "page_index": page,
                "begin_time": str(start),
                "end_time": str(end),
            },
        )
        data = payload.get("data")
        if payload.get("success") != 1 or not isinstance(data, dict):
            raise VendorUnavailableError("Eastmoney announcement response did not succeed")
        items, total = data.get("list"), data.get("total_hits")
        if not isinstance(items, list) or not isinstance(total, int) or total < 0:
            raise VendorUnavailableError("Eastmoney announcement schema changed")
        if data.get("page_index") != page or len(items) > PAGE_SIZE:
            raise VendorUnavailableError("Eastmoney announcement pagination changed")
        for item in items:
            if not isinstance(item, dict):
                raise VendorUnavailableError("Eastmoney malformed announcement")
            codes = item.get("codes")
            if not isinstance(codes, list) or not any(
                isinstance(code, dict)
                and code.get("stock_code") == stock.code
                and code.get("market_code") == stock.secid[0]
                for code in codes
            ):
                continue
            title, article = item.get("title"), item.get("art_code")
            notice = _publication_time(item.get("notice_date"))
            display = _publication_time(item.get("display_time"))
            indexed = _publication_time(item.get("eiTime"))
            if (
                not isinstance(title, str)
                or not isinstance(article, str)
                or not re.fullmatch(r"AN\d+", article)
                or notice is None
                or (item.get("display_time") and display is None)
                or (item.get("eiTime") and indexed is None)
            ):
                excluded_unknown += 1
                continue
            # A nominal midnight alone proves no intraday availability: next day.
            available = (
                max(notice, display, indexed or display)
                if display
                else max(notice + timedelta(days=1), indexed or notice)
            )
            if not start <= notice.date() <= end or available > cutoff:
                continue
            period = _report_period(title)
            if (period and _date(period) > end) or (reports_only and not period):
                continue
            if freq == "annual" and (not period or not period.endswith("-12-31")):
                continue
            if freq == "quarterly" and (not period or period.endswith("-12-31")):
                continue
            if article not in seen:
                seen.add(article)
                rows.append(
                    {
                        "NoticeDate": str(notice.date()),
                        "AvailableAt": available.isoformat(),
                        "ReportPeriodEnd": period or "unknown",
                        "Title": title,
                        "URL": f"https://data.eastmoney.com/notices/detail/{stock.code}/{article}.html",
                    }
                )
        if page * PAGE_SIZE >= total:
            break
        if len(items) < PAGE_SIZE:
            truncated = True
            break
        truncated = page == MAX_ANNOUNCEMENT_PAGES
    rows.sort(key=lambda row: row["AvailableAt"], reverse=True)
    header = (
        f"# Company announcement index for {stock.canonical}; source: Eastmoney/Choice public index\n"
        f"# Window: {start}..{end}; as_of cutoff: {cutoff.isoformat()}; retrieved: {_now().isoformat()}\n"
        "# ReportPeriodEnd is parsed only from explicit report titles, not its publication date.\n"
        "# Filter uses the latest of notice/display/index timestamps; nominal date-only items wait until next day.\n"
        "# Current index, not an archived as-of snapshot: later edits, removals and omissions cannot be reconstructed.\n"
        f"# Partial/truncated pagination: {truncated}; excluded unknown-time/schema items: {excluded_unknown}\n"
        "# Structured statements, financial metrics and domestic news/sentiment are DATA_UNAVAILABLE in phase 1.\n"
        "# Missing/empty announcements never mean no risks. Verify original disclosures at CNINFO/SSE/SZSE.\n"
    )
    if not rows:
        return (
            header
            + "DATA_UNAVAILABLE: no verifiable index entries in this window; coverage/absence is not established."
        )
    return header + pd.DataFrame(rows).to_csv(index=False)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    return get_announcements(ticker, start_date, end_date)


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    end = _date(curr_date) if curr_date else _now().date()
    return get_announcements(ticker, str(end - timedelta(days=400)), str(end), reports_only=True)


def _statement(ticker: str, freq: str, curr_date: str | None, title: str) -> str:
    end = _date(curr_date) if curr_date else _now().date()
    return (
        f"DATA_UNAVAILABLE: {title} numeric tables are not implemented; the following is only a report index.\n"
        + get_announcements(
            ticker, str(end - timedelta(days=400)), str(end), reports_only=True, freq=freq
        )
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, freq, curr_date, "balance sheet")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, freq, curr_date, "cash flow")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, freq, curr_date, "income statement")


def unavailable(topic: str) -> str:
    return (
        f"DATA_UNAVAILABLE: A-share phase 1 does not implement {topic}. "
        "This is missing evidence, not an absence of risks. Do not infer or fabricate values."
    )


def get_insider_transactions(ticker: str, curr_date: str | None = None) -> str:
    parse_ashare_symbol(ticker)
    return unavailable("domestic insider/major-holder transactions or institutional accounts")
