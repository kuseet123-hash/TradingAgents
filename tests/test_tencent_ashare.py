"""Fixed domestic units and closed routing; all data is synthetic and offline."""

import copy
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
import requests

from tradingagents.dataflows.ashare_market import sessions_between
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import tencent as tx

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(tx, "_now", lambda: datetime(2026, 10, 29, 12, tzinfo=tx.SHANGHAI))


@pytest.fixture
def rows():
    return json.loads((Path(__file__).parent / "fixtures/tencent_synthetic.json").read_text())[
        "rows"
    ]


def feed(monkeypatch, rows, symbol="sh603179", **extras):
    payload = {"code": 0, "data": {symbol: {"day": rows, **extras}}}
    fetch = Mock(return_value=payload)
    monkeypatch.setattr(tx, "_get_payload", fetch)
    return fetch


@pytest.mark.parametrize(("symbol", "raw"), [("603179", "sh603179"), ("000001", "sz000001")])
def test_fixed_units_including_sz000(monkeypatch, rows, symbol, raw):
    fetch = feed(monkeypatch, rows, raw)
    frame = tx.load_daily_bars(symbol, "2026-10-27", "2026-10-28")
    assert frame.iloc[-1]["Volume"] == 200000  # sz000 must also multiply lots by 100
    assert frame.iloc[-1]["VolumeLots"] == 2000
    assert frame.iloc[-1]["AmountTenThousandYuan"] == 240
    assert frame.iloc[-1]["Amount"] == 2400000  # source amount, not close * volume
    assert frame.iloc[-1]["Turnover"] == 0.025
    assert frame.attrs["verified_unit_rows"] == 2
    assert ",day,2026-10-27,2026-10-28,20," in fetch.call_args.args[0]["param"]
    assert fetch.call_args.args[0]["param"].startswith(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        "shares_instead_of_lots",
        "yuan_instead_of_wanyuan",
        "missing_amount",
        "nan_amount",
        "zero_amount",
        "unknown_shape",
        "bad_ohlc",
        "negative",
        "duplicate",
        "holiday",
        "all_zero",
    ],
)
def test_unverified_fields_fail_closed_without_unit_guessing(monkeypatch, rows, mutate):
    if mutate == "shares_instead_of_lots":
        rows[0][5] = "100000"
    elif mutate == "yuan_instead_of_wanyuan":
        rows[0][8] = "1100000"
    elif mutate == "missing_amount":
        rows[0] = rows[0][:6]
    elif mutate == "nan_amount":
        rows[0][8] = "NaN"
    elif mutate == "zero_amount":
        rows[0][8] = "0"
    elif mutate == "unknown_shape":
        rows[0].append("new field")
    elif mutate == "bad_ohlc":
        rows[0][1] = "100"
    elif mutate == "negative":
        rows[0][5] = "-1000"
    elif mutate == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutate == "holiday":
        extra = copy.deepcopy(rows[0])
        extra[0] = "2026-10-25"
        rows.append(extra)
    elif mutate == "all_zero":
        for row in rows:
            row[5] = row[8] = "0"
    feed(monkeypatch, rows)
    with pytest.raises(VendorUnavailableError, match="unverified|cannot be verified"):
        tx.load_daily_bars("603179", "2026-10-20", "2026-10-28")


def test_amount_is_never_synthesized_from_close(monkeypatch, rows):
    rows[0][8] = "105"  # source VWAP=10.5, different from close=11
    feed(monkeypatch, rows)
    frame = tx.load_daily_bars("603179", "2026-10-27", "2026-10-28")
    assert frame.iloc[0]["Amount"] == 1050000
    assert frame.iloc[0]["Amount"] != frame.iloc[0]["Close"] * frame.iloc[0]["Volume"]


def test_rounding_tolerance_cannot_hide_penny_price_unit_error():
    with pytest.raises(ValueError, match="unit contract fails"):
        tx._unit_check(0.01, 0.02, 10000, 1)


def test_today_and_future_rows_withheld_before_indicators(monkeypatch, rows):
    for day in ("2026-10-29", "2026-10-30"):
        extra = copy.deepcopy(rows[-1])
        extra[0] = day
        rows.append(extra)
    fetch = feed(monkeypatch, rows)
    frame = tx.load_daily_bars("603179", "2026-10-27", "2026-10-29")
    assert list(frame.Date.dt.strftime("%Y-%m-%d")) == ["2026-10-27", "2026-10-28"]
    assert "2026-10-28" in fetch.call_args.args[0]["param"]
    assert "2026-10-29" not in fetch.call_args.args[0]["param"]


@pytest.mark.parametrize("kind", ["empty", "stale", "adjusted_only", "wrong_symbol"])
def test_no_data_and_wrong_basis_are_not_prices(monkeypatch, rows, kind):
    if kind == "empty":
        feed(monkeypatch, [])
    elif kind == "stale":
        feed(monkeypatch, rows[:1])
    elif kind == "wrong_symbol":
        feed(monkeypatch, rows, "sz000001")
    else:
        monkeypatch.setattr(
            tx, "_get_payload", lambda _: {"code": 0, "data": {"sh603179": {"qfqday": rows}}}
        )
    error = NoMarketDataError if kind in {"empty", "stale"} else VendorUnavailableError
    with pytest.raises(error):
        tx.load_daily_bars("603179", "2026-10-27", "2026-10-28")


@pytest.mark.parametrize("adjustment", ["qfq", "hfq", "", "adjusted"])
def test_adjustment_refused_without_request(monkeypatch, adjustment):
    fetch = Mock()
    monkeypatch.setattr(tx, "_get_payload", fetch)
    with pytest.raises(ValueError, match="as-of vintage"):
        tx.load_daily_bars("603179", "2026-10-27", "2026-10-28", adjustment=adjustment)
    fetch.assert_not_called()


def test_unverified_star_board_units_and_holidays_do_not_request(monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(tx, "_get_payload", fetch)
    with pytest.raises(VendorUnavailableError, match="STAR-board"):
        tx.load_daily_bars("688001", "2026-10-27", "2026-10-28")
    with pytest.raises(NoMarketDataError, match="no completed sessions"):
        tx.load_daily_bars("000001", "2026-10-01", "2026-10-07")
    fetch.assert_not_called()


def session_response(monkeypatch, *, status=200, body="{}", error=None):
    session = MagicMock()
    session.get.return_value = Mock(status_code=status, text=body)
    if error:
        session.get.side_effect = error
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(tx.requests, "Session", factory)
    return session


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_restrictions_no_redirect_or_retry(monkeypatch, status):
    session = session_response(monkeypatch, status=status)
    with pytest.raises(VendorUnavailableError, match=f"HTTP {status}"):
        tx._get_payload({"_var": "kline_day2026"})
    assert session.get.call_count == 1
    assert session.get.call_args.kwargs["verify"] is True
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert session.mount.call_args.args[1].max_retries.total == 0


@pytest.mark.parametrize(
    "error", [requests.Timeout(), requests.exceptions.SSLError(), requests.ConnectionError()]
)
def test_network_and_tls_errors_do_not_retry(monkeypatch, error):
    session = session_response(monkeypatch, error=error)
    with pytest.raises(VendorUnavailableError):
        tx._get_payload({"_var": "kline_day2026"})
    assert session.get.call_count == 1


def test_domestic_public_feeds_never_read_netrc_credentials(monkeypatch):
    from tradingagents.dataflows.vendors import eastmoney

    forbidden = Mock(side_effect=AssertionError("netrc credentials were read"))
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", forbidden)
    prepared = []

    def send(session, request, **kwargs):
        prepared.append(request)
        return Mock(status_code=200, text='{"code":0,"data":{}}', json=lambda: {})

    monkeypatch.setattr(requests.Session, "send", send)
    tx._get_payload({"_var": "kline_day2026"})
    eastmoney._get_json(eastmoney.ANNOUNCEMENT_URL, {})
    assert len(prepared) == 2
    assert all("Authorization" not in request.headers for request in prepared)
    forbidden.assert_not_called()


@pytest.mark.parametrize("prefix", ["", "kline_day2026="])
def test_strict_json_or_expected_jsonp(monkeypatch, prefix):
    session_response(monkeypatch, body=prefix + '{"code":0,"data":{}};')
    assert tx._get_payload({"_var": "kline_day2026"}) == {"code": 0, "data": {}}


@pytest.mark.parametrize(
    "body", ["<html>challenge</html>", 'unknown={"code":0}', '{"code":1}', "[]"]
)
def test_unknown_payloads_are_unavailable(monkeypatch, body):
    session_response(monkeypatch, body=body)
    with pytest.raises(VendorUnavailableError):
        tx._get_payload({"_var": "kline_day2026"})


def test_price_indicator_snapshot_share_filtered_loader_and_values(monkeypatch):
    from tradingagents.agents.tools import (
        get_indicators,
        get_stock_data,
        get_verified_market_snapshot,
    )

    dates = sorted(
        sessions_between(pd.Timestamp("2025-01-02").date(), pd.Timestamp("2026-10-28").date())
    )
    rows = []
    for index, day in enumerate(dates):
        price = 10 + index / 100
        rows.append(
            [
                str(day),
                str(price),
                str(price + 0.1),
                str(price + 0.2),
                str(price - 0.2),
                "1000",
                "",
                "1",
                str((price + 0.05) * 10),
                "",
                "",
            ]
        )
    feed(monkeypatch, rows, "sz000001")
    with run_config({"ashare_data_enabled": True, "ashare_price_vendor": "tencent"}):
        prices = get_stock_data.func("000001", "2026-10-27", "2026-10-28")
        indicators = get_indicators.func("000001", "rsi", "2026-10-30", 3, "2026-10-28")
        snapshot = get_verified_market_snapshot.func("000001", "2026-10-30", 2, "2026-10-28")
    assert (
        "source: Tencent" in prices
        and "source: Tencent" in indicators
        and "source: Tencent" in snapshot
    )
    assert "Latest trading row used: 2026-10-28" in snapshot
    assert "2026-10-29," not in snapshot and "2026-10-30," not in snapshot
    assert "| close_200_sma | N/A" not in snapshot
    assert "2026-10-28: 100" in indicators and "| rsi | 100 |" in snapshot
    assert prices.splitlines()[-1] == snapshot.split("\n\n| Indicator")[0].splitlines()[-1]


@pytest.mark.parametrize(
    "failure", [VendorUnavailableError("unit check failed"), NoMarketDataError("000001")]
)
def test_tencent_errors_never_call_other_vendors(monkeypatch, failure):
    from tradingagents.dataflows import router

    blocked = Mock(side_effect=AssertionError("fallback or foreign vendor called"))
    for method in ("get_stock_data", "get_indicators", "get_verified_market_snapshot"):
        monkeypatch.setitem(router.VENDOR_METHODS[method], "tencent", Mock(side_effect=failure))
        for vendor in ("eastmoney", "yfinance", "sec_edgar", "alpha_vantage"):
            monkeypatch.setitem(router.VENDOR_METHODS[method], vendor, blocked)
    with run_config({"ashare_data_enabled": True, "ashare_price_vendor": "tencent"}):
        for method in ("get_stock_data", "get_indicators", "get_verified_market_snapshot"):
            result = router.route_to_vendor(method, "000001")
            assert result.startswith(("DATA_UNAVAILABLE", "NO_DATA_AVAILABLE"))
    blocked.assert_not_called()


def test_announcement_vendor_remains_domestic_and_independent(monkeypatch):
    from tradingagents.dataflows import router

    news = Mock(return_value="DOMESTIC DISCLOSURE INDEX ONLY")
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "eastmoney", news)
    with run_config({"ashare_data_enabled": True, "ashare_price_vendor": "tencent"}):
        assert (
            router.route_to_vendor("get_news", "000001", "2026-10-01", "2026-10-28")
            == "DOMESTIC DISCLOSURE INDEX ONLY"
        )
    news.assert_called_once()


@pytest.mark.parametrize("vendor", ["yfinance", "sec_edgar", "tencent,eastmoney", "default"])
def test_unknown_or_multiple_price_sources_fail_closed(vendor):
    from tradingagents.dataflows import router

    with run_config({"ashare_data_enabled": True, "ashare_price_vendor": vendor}):
        assert "no automatic vendor fallback" in router.route_to_vendor("get_stock_data", "000001")
