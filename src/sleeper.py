"""Sleeper API client.

Sleeper's read endpoints are public and unauthenticated, which is the entire
reason this project is a weekend of work rather than a month. Three of the four
endpoints used here are documented; the projections endpoint is not, so it is
wrapped defensively and the pipeline degrades to ADP-derived values if it moves.
"""

from __future__ import annotations

from .netcache import FetchError, get_json

BASE = "https://api.sleeper.app"

SCORING_KEY = {"ppr": "pts_ppr", "half_ppr": "pts_half_ppr", "std": "pts_std"}

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


def get_state() -> dict:
    return get_json(f"{BASE}/v1/state/nfl", ttl_hours=1)


def get_players(ttl_hours: float = 24) -> dict:
    """The full player dictionary (~14 MB). Cached hard -- it changes slowly."""
    return get_json(f"{BASE}/v1/players/nfl", ttl_hours=ttl_hours, timeout=180)


def get_projections(season: str, scoring: str = "ppr", ttl_hours: float = 6) -> dict:
    """player_id -> season projected fantasy points, for the given scoring.

    Undocumented endpoint. Returns {} on failure so callers can fall back.
    """
    key = SCORING_KEY.get(scoring, "pts_ppr")
    out: dict[str, float] = {}
    for pos in FANTASY_POSITIONS:
        url = (
            f"{BASE}/projections/nfl/{season}"
            f"?season_type=regular&position[]={pos}&order_by={key}"
        )
        try:
            rows = get_json(url, ttl_hours=ttl_hours, timeout=90)
        except FetchError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            pid = row.get("player_id")
            stats = row.get("stats") or {}
            pts = stats.get(key)
            if pid and isinstance(pts, (int, float)) and pts > 0:
                # Keep the best value if a player somehow appears twice.
                if pts > out.get(str(pid), 0.0):
                    out[str(pid)] = float(pts)
    return out


def get_user(username: str) -> dict:
    return get_json(f"{BASE}/v1/user/{username}", ttl_hours=1)


def get_user_leagues(user_id: str, season: str) -> list:
    return get_json(f"{BASE}/v1/user/{user_id}/leagues/nfl/{season}", ttl_hours=0.25)


def get_league_drafts(league_id: str) -> list:
    return get_json(f"{BASE}/v1/league/{league_id}/drafts", ttl_hours=0.25)


def get_user_drafts(user_id: str, season: str) -> list:
    """All drafts for a user, including standalone mock drafts.

    Mocks are not attached to a league, so walking leagues -> drafts misses
    them entirely. This is the only endpoint that finds them.
    """
    drafts = get_json(f"{BASE}/v1/user/{user_id}/drafts/nfl/{season}", ttl_hours=0)
    return drafts if isinstance(drafts, list) else []


def get_draft(draft_id: str) -> dict:
    return get_json(f"{BASE}/v1/draft/{draft_id}", ttl_hours=0)


def get_draft_picks(draft_id: str) -> list:
    """Live pick feed. Never cached."""
    picks = get_json(f"{BASE}/v1/draft/{draft_id}/picks", ttl_hours=0, timeout=20)
    return picks if isinstance(picks, list) else []


def resolve_draft_id(username: str, season: str) -> tuple[str | None, str]:
    """Find the user's most relevant draft for the season.

    Returns (draft_id, human readable note).
    """
    try:
        user = get_user(username)
    except FetchError as exc:
        return None, f"could not look up user {username!r}: {exc}"
    if not user or not user.get("user_id"):
        return None, f"no Sleeper user named {username!r}"

    # No leagues is not fatal -- a mock draft needs no league at all.
    try:
        leagues = get_user_leagues(user["user_id"], season) or []
    except FetchError:
        leagues = []

    found = []
    for lg in leagues:
        try:
            drafts = get_league_drafts(lg["league_id"]) or []
        except FetchError:
            continue
        for d in drafts:
            found.append((lg.get("name", "?"), d))

    # Standalone mock drafts live outside any league.
    try:
        for d in get_user_drafts(user["user_id"], season):
            if not any(d["draft_id"] == existing["draft_id"] for _, existing in found):
                found.append(("mock draft", d))
    except FetchError:
        pass

    if not found:
        return None, "no drafts found in any league"

    # Prefer an in-progress draft, then a pre-draft one, then the newest.
    order = {"drafting": 0, "paused": 1, "pre_draft": 2, "complete": 3}
    found.sort(key=lambda t: (order.get(t[1].get("status"), 9), -(t[1].get("start_time") or 0)))
    name, draft = found[0]
    return draft["draft_id"], f"{name} ({draft.get('status')})"


def my_slot_from_draft(draft: dict, user_id: str) -> int | None:
    """Read the user's draft slot out of the draft object's draft_order map."""
    order = draft.get("draft_order") or {}
    slot = order.get(user_id)
    return int(slot) if slot else None
