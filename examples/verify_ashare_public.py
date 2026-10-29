"""Small public-feed check. No LLM, key, .env loading, persistence or price redistribution."""

import argparse
import csv
import io
import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import requests

# Deliberately isolated from any deployed application's configuration/secrets.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from tradingagents.dataflows.config import run_config  # noqa: E402
from tradingagents.dataflows.router import route_to_vendor  # noqa: E402
from tradingagents.dataflows.vendors import eastmoney, tencent  # noqa: E402


def csv_rows(text):
    """Read only a tool's OHLCV CSV, not its following indicator table."""
    lines = text.splitlines()
    begin = next((i for i, line in enumerate(lines) if line.startswith("Date,Open,")), None)
    if begin is None:
        return []
    selected = []
    for line in lines[begin:]:
        if not line or line.startswith("|"):
            break
        selected.append(line)
    return list(csv.DictReader(io.StringIO("\n".join(selected))))


def verify_derived(symbol, as_of, rows, prices):
    indicators = route_to_vendor("get_indicators", symbol, "rsi", as_of, 7)
    snapshot = route_to_vendor("get_verified_market_snapshot", symbol, as_of, 30)
    snapshot_rows = csv_rows(snapshot)
    latest = rows[-1] if rows else {}
    snap_latest = snapshot_rows[-1] if snapshot_rows else {}
    prefix = latest.get("Date", "") + ": "
    value = next(
        (line[len(prefix) :] for line in indicators.splitlines() if line.startswith(prefix)), "N/A"
    )
    return {
        "indicator_available": value != "N/A" and not value.startswith("N/A:"),
        "indicator_equals_snapshot": f"| rsi | {value} |" in snapshot,
        "snapshot_available": bool(snapshot_rows),
        "same_normalized_latest_bar": bool(latest)
        and all(
            latest.get(key) == snap_latest.get(key)
            for key in ("Date", "Open", "High", "Low", "Close", "Volume", "Amount")
        ),
        "same_source": len(
            {
                next((line for line in text.splitlines() if line.startswith("# A-share")), "")
                for text in (prices, indicators, snapshot)
            }
        )
        == 1,
        "sma200_available": "| close_200_sma |" in snapshot
        and "| close_200_sma | N/A" not in snapshot,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="at most two SH/SZ equity codes")
    parser.add_argument(
        "--as-of", default=str(datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1))
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--vendor", choices=("eastmoney", "tencent"), default="eastmoney")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also verify RSI and the snapshot (up to 730 calendar days)",
    )
    args = parser.parse_args()
    symbols = args.symbol or ["603179", "000001"]
    if len(symbols) > 2:
        parser.error("this connectivity check is limited to two instruments")
    end = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    start = args.start or str(end - timedelta(days=14))
    reports, request_log = [], []
    request = requests.Session.request

    def audited_request(session, method, url, **kwargs):
        allowed = {eastmoney.PRICE_URL, eastmoney.ANNOUNCEMENT_URL, tencent.PRICE_URL}
        if url not in allowed:
            raise RuntimeError("Verification refused an unexpected data destination")
        entry = {
            "source": "tencent" if url == tencent.PRICE_URL else "eastmoney",
            "kind": "announcements" if url == eastmoney.ANNOUNCEMENT_URL else "daily",
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "http_status": None,
        }
        started = time.monotonic()
        try:
            response = request(session, method, url, **kwargs)
            entry["http_status"] = response.status_code
            return response
        except requests.RequestException as exc:
            entry["error_type"] = type(exc).__name__
            raise
        finally:
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
            request_log.append(entry)

    with (
        run_config({"ashare_data_enabled": True, "ashare_price_vendor": args.vendor}),
        patch.object(requests.Session, "request", audited_request),
    ):
        for symbol in symbols:
            prices = route_to_vendor("get_stock_data", symbol, start, str(end))
            disclosures = route_to_vendor("get_news", symbol, start, str(end))
            rows = csv_rows(prices)
            derived = verify_derived(symbol, str(end), rows, prices) if args.full and rows else {}
            reports.append(
                {
                    "symbol": symbol,
                    "price_vendor": args.vendor,
                    "window": [start, str(end)],
                    "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "prices": "ok" if rows else prices,
                    "completed_rows": len(rows),
                    "dates": [row["Date"] for row in rows],
                    "last_session": rows[-1]["Date"] if rows else None,
                    "units": {"price": "CNY yuan/share", "volume": "shares", "amount": "CNY yuan"},
                    "adjustment": "none",
                    "today_withheld": True,
                    "announcement_index": "reachable"
                    if disclosures.startswith("# Company announcement index")
                    else disclosures,
                    "index_entries_available": disclosures.startswith(
                        "# Company announcement index"
                    )
                    and "no verifiable index entries" not in disclosures,
                    "historical_snapshot_guaranteed": False,
                    "derived_checks": derived,
                }
            )
    print(
        json.dumps({"instruments": reports, "requests": request_log}, ensure_ascii=False, indent=2)
    )
    return (
        0
        if all(
            report["prices"] == "ok"
            and report["announcement_index"] == "reachable"
            and (
                not args.full
                or (report["derived_checks"] and all(report["derived_checks"].values()))
            )
            for report in reports
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
