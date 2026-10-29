"""Small public-feed check. No LLM, key, .env loading, persistence or price redistribution."""

import argparse
import csv
import io
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Deliberately isolated from any deployed application's configuration/secrets.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from tradingagents.dataflows.config import run_config  # noqa: E402
from tradingagents.dataflows.router import route_to_vendor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="at most two SH/SZ equity codes")
    parser.add_argument(
        "--as-of", default=str(datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1))
    )
    parser.add_argument("--start", default=None)
    args = parser.parse_args()
    symbols = args.symbol or ["603179", "000001"]
    if len(symbols) > 2:
        parser.error("this connectivity check is limited to two instruments")
    end = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    start = args.start or str(end - timedelta(days=14))
    reports = []
    with run_config({"ashare_data_enabled": True}):
        for symbol in symbols:
            prices = route_to_vendor("get_stock_data", symbol, start, str(end))
            disclosures = route_to_vendor("get_news", symbol, start, str(end))
            lines = "\n".join(line for line in prices.splitlines() if not line.startswith("#"))
            rows = (
                list(csv.DictReader(io.StringIO(lines))) if prices.startswith("# A-share") else []
            )
            reports.append(
                {
                    "symbol": symbol,
                    "window": [start, str(end)],
                    "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "prices": "ok" if rows else prices,
                    "completed_rows": len(rows),
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
                }
            )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return (
        0
        if all(
            report["prices"] == "ok" and report["announcement_index"] == "reachable"
            for report in reports
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
