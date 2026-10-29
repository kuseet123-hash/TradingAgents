"""Phase-1 domestic chain: synthetic responses, no keys, LLM or live network."""

import copy
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from tradingagents.dataflows.ashares import parse_ashare_symbol
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors import eastmoney as em

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic():
    return json.loads((Path(__file__).parent / "fixtures" / "ashare_synthetic.json").read_text())


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(em, "_now", lambda: datetime(2026, 10, 29, 12, tzinfo=em.SHANGHAI))


@pytest.mark.parametrize(
    ("raw", "canonical", "secid"),
    [
        ("603179", "603179.SS", "1.603179"),
        (" sh603179 ", "603179.SS", "1.603179"),
        ("603179.SH", "603179.SS", "1.603179"),
        ("603179.SS", "603179.SS", "1.603179"),
        ("000001", "000001.SZ", "0.000001"),
        ("sz000001", "000001.SZ", "0.000001"),
        ("300750.SZ", "300750.SZ", "0.300750"),
        ("688001.SH", "688001.SS", "1.688001"),
    ],
)
def test_symbol_namespace_and_normalization(raw, canonical, secid):
    stock = parse_ashare_symbol(raw)
    assert stock.canonical == canonical
    assert stock.secid == secid
    assert normalize_symbol(raw) == canonical


@pytest.mark.parametrize(
    "raw",
    [
        "603179.SZ",
        "000001.SS",
        "SH000001",
        "SZ603179",
        "SH603179.SZ",
        "900901.SH",
        "200001.SZ",
        "510300.SS",
        "830001.BJ",
        "689009.SH",
        "60317",
        "AAPL",
        "603179/../../",
        "999999",
        None,
    ],
)
def test_unknown_conflicting_and_non_equity_symbols_rejected(raw):
    with pytest.raises(ValueError):
        parse_ashare_symbol(raw)


def price_response(monkeypatch, synthetic):
    fetch = Mock(return_value=synthetic["prices"])
    monkeypatch.setattr(em, "_get_json", fetch)
    return fetch


def test_price_units_identity_and_unadjusted_protocol(monkeypatch, synthetic):
    fetch = price_response(monkeypatch, synthetic)
    frame = em.load_daily_bars("603179", "2026-10-27", "2026-10-28")
    assert frame.iloc[-1]["Open"] == 11
    assert frame.iloc[-1]["High"] == 13
    assert frame.iloc[-1]["Low"] == 10
    assert frame.iloc[-1]["Close"] == 12
    assert frame.iloc[-1]["VolumeLots"] == 2000
    assert frame.iloc[-1]["Volume"] == 200000
    assert frame.iloc[-1]["Amount"] == 2400000.75  # yuan, no *10000
    assert frame.attrs["adjustment"] == "none"
    params = fetch.call_args.args[1]
    assert params["secid"] == "1.603179" and params["fqt"] == "0"
    assert params["beg"] == "20261027" and params["end"] == "20261028"
    output = em.get_stock_data("603179", "2026-10-27", "2026-10-28")
    assert "Volume: shares" in output and "Amount: CNY yuan" in output
    assert "no archived vintage" in output


@pytest.mark.parametrize("adjustment", ["qfq", "hfq", "adjusted", ""])
def test_adjusted_prices_refused_before_request(monkeypatch, adjustment):
    fetch = Mock()
    monkeypatch.setattr(em, "_get_json", fetch)
    with pytest.raises(ValueError, match="as-of vintage"):
        em.load_daily_bars("603179", "2026-10-27", "2026-10-28", adjustment=adjustment)
    fetch.assert_not_called()


@pytest.mark.parametrize("hour", [9, 12, 15, 18])
def test_current_day_withheld_even_after_close(monkeypatch, synthetic, hour):
    monkeypatch.setattr(em, "_now", lambda: datetime(2026, 10, 29, hour, tzinfo=em.SHANGHAI))
    synthetic["prices"]["data"]["klines"].append("2026-10-29,12,13,14,11,3000,3900000,0,0,0,0")
    fetch = price_response(monkeypatch, synthetic)
    frame = em.load_daily_bars("603179", "2026-10-27", "2026-10-29")
    assert str(frame.iloc[-1]["Date"].date()) == "2026-10-28"
    assert fetch.call_args.args[1]["end"] == "20261028"


def test_shanghai_date_independent_of_host_timezone(monkeypatch, synthetic):
    monkeypatch.setattr(em, "_now", lambda: datetime(2026, 10, 29, 0, 1, tzinfo=em.SHANGHAI))
    price_response(monkeypatch, synthetic)
    frame = em.load_daily_bars("603179", "2026-10-28", "2026-10-29")
    assert str(frame.iloc[-1]["Date"].date()) == "2026-10-28"


def test_national_holiday_and_weekend_make_no_requests(monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(em, "_get_json", fetch)
    for start, end in [("2026-10-01", "2026-10-07"), ("2026-10-10", "2026-10-11")]:
        with pytest.raises(NoMarketDataError, match="no completed sessions"):
            em.load_daily_bars("000001", start, end)
    fetch.assert_not_called()
    assert em._sessions(em._date("2026-09-30"), em._date("2026-10-08")) == {
        em._date("2026-09-30"),
        em._date("2026-10-08"),
    }


def test_stale_rows_not_forward_filled_across_holiday(monkeypatch, synthetic):
    synthetic["prices"]["data"]["klines"] = ["2026-09-30,10,11,12,9,1000,1100000,0,0,0,0"]
    price_response(monkeypatch, synthetic)
    with pytest.raises(NoMarketDataError, match="expected session 2026-10-08"):
        em.load_daily_bars("603179", "2026-09-30", "2026-10-08")


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "null",
        "wrong_code",
        "wrong_market",
        "malformed",
        "nan",
        "negative",
        "bad_ohlc",
        "duplicate",
        "non_session",
        "rc",
    ],
)
def test_empty_and_corrupt_source_data(monkeypatch, synthetic, kind):
    data = synthetic["prices"]["data"]
    if kind == "empty":
        data["klines"] = []
    elif kind == "null":
        synthetic["prices"]["data"] = None
    elif kind == "wrong_code":
        data["code"] = "600000"
    elif kind == "wrong_market":
        data["market"] = 0
    elif kind == "malformed":
        data["klines"] = ["bad,row"]
    elif kind == "nan":
        data["klines"][0] = data["klines"][0].replace(",10,", ",NaN,", 1)
    elif kind == "negative":
        data["klines"][0] = data["klines"][0].replace(",1000,", ",-1000,", 1)
    elif kind == "bad_ohlc":
        data["klines"][0] = "2026-10-27,100,11,12,9,1000,1,0,0,0,0"
    elif kind == "duplicate":
        data["klines"].append(data["klines"][0])
    elif kind == "non_session":
        data["klines"].append("2026-10-25,10,11,12,9,1000,1,0,0,0,0")
    elif kind == "rc":
        synthetic["prices"]["rc"] = -1
    price_response(monkeypatch, synthetic)
    error = NoMarketDataError if kind in {"empty", "null"} else VendorUnavailableError
    with pytest.raises(error):
        em.load_daily_bars("603179", "2026-10-20", "2026-10-28")


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_access_failures_stop_without_retry(monkeypatch, status):
    fetch = Mock(return_value=Mock(status_code=status))
    monkeypatch.setattr(em.requests, "get", fetch)
    with pytest.raises(VendorUnavailableError, match=f"HTTP {status}"):
        em._get_json(em.PRICE_URL, {})
    assert fetch.call_count == 1
    assert fetch.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("failure", [requests.Timeout(), ValueError("HTML challenge")])
def test_timeout_and_invalid_json_are_vendor_errors(monkeypatch, failure):
    response = Mock(status_code=200)
    response.json.side_effect = failure
    monkeypatch.setattr(em.requests, "get", Mock(return_value=response))
    with pytest.raises(VendorUnavailableError):
        em._get_json(em.PRICE_URL, {})


def announcement_response(monkeypatch, rows, total=None):
    fetch = Mock(
        return_value={
            "success": 1,
            "data": {
                "list": rows,
                "total_hits": len(rows) if total is None else total,
                "page_index": 1,
            },
        }
    )
    monkeypatch.setattr(em, "_get_json", fetch)
    return fetch


def test_report_period_does_not_make_undisclosed_report_public(monkeypatch, synthetic):
    announcement_response(monkeypatch, [synthetic["announcement"]])
    before = em.get_announcements("603179", "2026-03-01", "2026-03-31")
    assert "合成测试公司" not in before
    assert "DATA_UNAVAILABLE" in before
    after = em.get_announcements("603179", "2026-03-01", "2026-04-01")
    assert "2025-12-31" in after
    assert "2026-04-01T00:00:00+08:00" in after
    assert "AN202604010000000001" in after
    assert "not an archived as-of snapshot" in after


def test_post_cutoff_reindex_or_correction_withheld(monkeypatch, synthetic):
    row = synthetic["announcement"]
    row["eiTime"] = "2026-04-03 09:01:00:000"
    announcement_response(monkeypatch, [row])
    assert "合成测试公司" not in em.get_announcements("603179", "2026-03-01", "2026-04-02")


def test_date_only_disclosure_waits_until_next_day(monkeypatch, synthetic):
    row = synthetic["announcement"]
    row.pop("display_time")
    row.pop("eiTime")
    announcement_response(monkeypatch, [row])
    same_day = em.get_announcements("603179", "2026-03-01", "2026-04-01")
    next_day = em.get_announcements("603179", "2026-03-01", "2026-04-02")
    assert "合成测试公司" not in same_day and "合成测试公司" in next_day


def test_unknown_publication_time_and_other_issuer_excluded(monkeypatch, synthetic):
    unknown = copy.deepcopy(synthetic["announcement"])
    unknown["notice_date"] = "unavailable"
    other = copy.deepcopy(synthetic["announcement"])
    other["codes"][0]["stock_code"] = "600000"
    announcement_response(monkeypatch, [unknown, other])
    result = em.get_announcements("603179", "2026-03-01", "2026-04-02")
    assert "合成测试公司" not in result
    assert "excluded unknown-time/schema items: 1" in result


def test_live_intraday_cutoff_does_not_include_evening_release(monkeypatch, synthetic):
    row = synthetic["announcement"]
    row.update(
        notice_date="2026-10-29 00:00:00",
        display_time="2026-10-29 18:00:00:000",
        eiTime="2026-10-29 18:01:00:000",
    )
    announcement_response(monkeypatch, [row])
    assert "合成测试公司" not in em.get_announcements("603179", "2026-10-01", "2026-10-29")


def test_empty_index_never_means_no_risk(monkeypatch):
    announcement_response(monkeypatch, [])
    result = em.get_news("000001", "2026-10-01", "2026-10-28")
    assert "DATA_UNAVAILABLE" in result
    assert "never mean no risks" in result
    assert "coverage/absence is not established" in result


def test_bounded_pagination_marks_partial_results(monkeypatch, synthetic):
    def reply(url, params):
        rows = []
        for item in range(em.PAGE_SIZE):
            row = copy.deepcopy(synthetic["announcement"])
            row["art_code"] = f"AN20260401{params['page_index']:02}{item:04}"
            rows.append(row)
        return {
            "success": 1,
            "data": {"list": rows, "total_hits": 999, "page_index": params["page_index"]},
        }

    fetch = Mock(side_effect=reply)
    monkeypatch.setattr(em, "_get_json", fetch)
    result = em.get_announcements("603179", "2026-03-01", "2026-04-02")
    assert fetch.call_count == em.MAX_ANNOUNCEMENT_PAGES
    assert "Partial/truncated pagination: True" in result


def test_structured_financial_values_explicitly_unavailable(monkeypatch, synthetic):
    announcement_response(monkeypatch, [synthetic["announcement"]])
    result = em.get_balance_sheet("603179", "annual", "2026-04-02")
    assert result.startswith("DATA_UNAVAILABLE")
    assert "only a report index" in result and "2025-12-31" in result
    with pytest.raises(ValueError, match="freq"):
        em.get_balance_sheet("603179", "monthly", "2026-04-02")


def test_indicator_warmup_and_snapshot_use_same_domestic_prices(monkeypatch, synthetic):
    price_response(monkeypatch, synthetic)
    output = em.build_verified_market_snapshot("603179", "2026-10-28")
    assert "2026-10-28" in output and "VolumeLots" in output
    assert "N/A: insufficient trading rows for warmup" in output
    assert "source: Eastmoney" in output
    with pytest.raises(ValueError, match="Unsupported"):
        em.get_indicators("603179", "fake", "2026-10-28", 3)


def test_ashare_router_and_snapshot_never_fall_back(monkeypatch, synthetic):
    from tradingagents.agents import tools
    from tradingagents.dataflows import router

    monkeypatch.setattr(em, "_get_json", Mock(side_effect=VendorUnavailableError("HTTP 429")))
    forbidden = Mock(side_effect=AssertionError("foreign fallback called"))
    for method in ("get_stock_data", "get_balance_sheet", "get_news"):
        monkeypatch.setitem(router.VENDOR_METHODS[method], "yfinance", forbidden)
        monkeypatch.setitem(router.VENDOR_METHODS[method], "alpha_vantage", forbidden)
        if "sec_edgar" in router.VENDOR_METHODS[method]:
            monkeypatch.setitem(router.VENDOR_METHODS[method], "sec_edgar", forbidden)
    monkeypatch.setattr(tools, "build_verified_market_snapshot", forbidden)
    with run_config({"ashare_data_enabled": True}):
        outputs = [
            router.route_to_vendor("get_stock_data", "603179", "2026-10-27", "2026-10-28"),
            router.route_to_vendor("get_balance_sheet", "000001", "annual", "2026-10-28"),
            router.route_to_vendor("get_news", "603179", "2026-10-01", "2026-10-28"),
            tools.get_verified_market_snapshot.func("603179", "2026-10-28"),
            router.route_to_vendor("get_stock_data", "603179.SZ", "2026-10-27", "2026-10-28"),
            router.route_to_vendor("get_stock_data", "SH603179.SZ", "2026-10-27", "2026-10-28"),
        ]
    assert all(result.startswith("DATA_UNAVAILABLE") for result in outputs)
    forbidden.assert_not_called()


def test_empty_domestic_data_never_calls_foreign_vendors(monkeypatch):
    from tradingagents.dataflows import router

    empty = Mock(side_effect=NoMarketDataError("603179", "603179.SS", "empty synthetic data"))
    forbidden = Mock(side_effect=AssertionError("foreign fallback called"))
    for method in (
        "get_stock_data",
        "get_balance_sheet",
        "get_news",
        "get_verified_market_snapshot",
    ):
        monkeypatch.setitem(router.VENDOR_METHODS[method], "eastmoney", empty)
        for vendor in ("sec_edgar", "yfinance", "alpha_vantage"):
            monkeypatch.setitem(router.VENDOR_METHODS[method], vendor, forbidden)
    with run_config({"ashare_data_enabled": True}):
        for method in (
            "get_stock_data",
            "get_balance_sheet",
            "get_news",
            "get_verified_market_snapshot",
        ):
            assert router.route_to_vendor(method, "603179").startswith("NO_DATA_AVAILABLE")
    forbidden.assert_not_called()


def test_unsupported_sources_identity_and_returns_make_no_foreign_requests(monkeypatch):
    from tradingagents.agents import context
    from tradingagents.agents.analysts import sentiment_analyst as sentiment
    from tradingagents.dataflows import router
    from tradingagents.memory import settlement

    forbidden = Mock(side_effect=AssertionError("unexpected external call"))
    monkeypatch.setattr(context, "get_company_profile", forbidden)
    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", forbidden)
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", forbidden)
    monkeypatch.setattr(sentiment, "jev_screen", forbidden)
    monkeypatch.setattr(sentiment, "bind_structured", lambda *a: None)
    monkeypatch.setattr(settlement, "get_closes", forbidden)
    with run_config({"ashare_data_enabled": True}):
        assert context.resolve_instrument_identity("603179") == {
            "exchange": "SH",
            "quote_type": "EQUITY",
        }
        assert "read-only research" in context.build_instrument_context("603179")
        node = sentiment.create_sentiment_analyst(Mock())
        assert "DATA_UNAVAILABLE" in node({"company_of_interest": "603179"})["sentiment_report"]
        for method, args in [
            ("get_global_news", ("2026-10-28",)),
            ("get_macro_indicators", ("cpi", "2026-10-28")),
            ("get_prediction_markets", ("China",)),
            ("get_insider_transactions", ("603179",)),
        ]:
            assert router.route_to_vendor(method, *args).startswith("DATA_UNAVAILABLE")
        with pytest.raises(VendorUnavailableError, match="return scoring unavailable"):
            settlement.fetch_returns("603179", "2026-01-01")
    forbidden.assert_not_called()


def test_us_routing_and_disabled_mode_preserved(monkeypatch):
    from tradingagents.dataflows import router

    foreign = Mock(return_value="EXISTING US DATA")
    monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "yfinance", foreign)
    with run_config({"ashare_data_enabled": True}):
        assert (
            router.route_to_vendor("get_stock_data", "AAPL", "2026-10-27", "2026-10-28")
            == "EXISTING US DATA"
        )
    with run_config({"ashare_data_enabled": False}):
        assert (
            router.route_to_vendor("get_stock_data", "603179.SS", "2026-10-27", "2026-10-28")
            == "EXISTING US DATA"
        )


def test_tool_clamps_nonpadded_future_date_before_domestic_call(monkeypatch):
    from tradingagents.agents import tools
    from tradingagents.dataflows import router

    seen = Mock(return_value="SYNTHETIC")
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "eastmoney", seen)
    with run_config({"ashare_data_enabled": True}):
        assert tools.get_news.func("603179", "2026-7-1", "2026-7-8", "2026-07-05") == "SYNTHETIC"
    assert seen.call_args.args == ("603179", "2026-07-01", "2026-07-05")
