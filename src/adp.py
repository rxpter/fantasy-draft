"""Average draft position from FantasyFootballCalculator.

FFC exposes ADP broken out by league size, which matters more than it sounds:
14-team ADP is materially different from the 12-team ADP that most public
rankings quote. It also ships a per-player stdev (the scarcity model needs it)
and bye weeks, which Sleeper's player file leaves null this time of year.
"""

from __future__ import annotations

import re
import unicodedata

from .netcache import FetchError, get_json

FFC = "https://fantasyfootballcalculator.com/api/v1/adp"

SCORING_PATH = {"ppr": "ppr", "half_ppr": "half-ppr", "std": "standard"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> tuple[str, ...]:
    """Fold a display name down to comparable tokens."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    tokens = [t for t in re.sub(r"[^a-z ]", "", s).split() if t and t not in _SUFFIXES]
    return tuple(tokens)


def name_key(name: str, position: str) -> tuple | None:
    toks = normalize_name(name)
    if not toks:
        return None
    return (toks[0], toks[-1], position)


def fetch_adp(teams: int, season: str, scoring: str = "ppr", ttl_hours: float = 6) -> list[dict]:
    """Fetch ADP, falling back to the previous season if the new one is empty."""
    path = SCORING_PATH.get(scoring, "ppr")
    url = f"{FFC}/{path}?teams={teams}&year={season}"
    try:
        data = get_json(url, ttl_hours=ttl_hours, timeout=40)
    except FetchError:
        return []
    players = (data or {}).get("players") or []
    return players


def build_adp_index(rows: list[dict]) -> dict:
    """(first, last, pos) -> adp row, plus DEF keyed by team."""
    idx: dict = {}
    for r in rows:
        pos = r.get("position") or ""
        if pos == "PK":
            pos = "K"
        if pos == "DEF":
            idx[("__def__", r.get("team"))] = r
            continue
        key = name_key(r.get("name", ""), pos)
        if key and key not in idx:
            idx[key] = r
    return idx


def lookup(idx: dict, name: str, position: str, team: str | None) -> dict | None:
    if position == "DEF":
        return idx.get(("__def__", team))
    return idx.get(name_key(name, position))
