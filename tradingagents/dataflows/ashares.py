"""Syntactic mainland equity identification; no company lookup or credentials."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SYMBOL = re.compile(r"(?:(SH|SZ))?(\d{6})(?:\.(SH|SS|SZ))?\Z")
_SH = re.compile(r"(?:60[0135]|688)\d{3}\Z")
_SZ = re.compile(r"(?:00[0-3]|30[01])\d{3}\Z")


@dataclass(frozen=True)
class AShareSymbol:
    code: str
    exchange: str

    @property
    def canonical(self) -> str:
        return f"{self.code}.{'SS' if self.exchange == 'SH' else 'SZ'}"

    @property
    def secid(self) -> str:
        return f"{1 if self.exchange == 'SH' else 0}.{self.code}"


def parse_ashare_symbol(raw: str) -> AShareSymbol:
    """Recognize SH/SZ ordinary equities, rejecting conflicting exchange hints.

    A bare 000001 means the SZ equity, never the Shanghai index. Funds, bonds,
    B shares, Beijing listings, CDRs and unknown prefixes are outside phase 1.
    This identifies an instrument namespace, not whether the code is listed.
    """
    match = _SYMBOL.fullmatch(raw.strip().upper()) if isinstance(raw, str) else None
    if match is None:
        raise ValueError(f"Unsupported A-share symbol: {raw!r}")
    prefix, code, suffix = match.groups()
    exchange = "SH" if _SH.fullmatch(code) else "SZ" if _SZ.fullmatch(code) else None
    hints = {"SH" if hint == "SS" else hint for hint in (prefix, suffix) if hint}
    if exchange is None or hints - {exchange}:
        raise ValueError(f"Unsupported A-share code or conflicting exchange: {raw!r}")
    return AShareSymbol(code, exchange)


def is_mainland_symbol(raw: str) -> bool:
    """Include unsupported mainland namespaces so mode never falls into US data."""
    if not isinstance(raw, str):
        return False
    value = raw.strip().upper()
    return bool(re.fullmatch(r"(?:SH|SZ)?\d{6}(?:\.(?:SH|SS|SZ|BJ))?", value))


def use_ashare_data(raw: str | None = None) -> bool:
    from tradingagents.dataflows.config import get_config

    return bool(get_config().get("ashare_data_enabled")) and (
        raw is None or is_mainland_symbol(raw)
    )
