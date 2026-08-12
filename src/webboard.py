"""Local web board.

Serves a single self-contained page on localhost plus a `/state.json` endpoint
the page polls. Uses only `http.server` from the standard library -- the whole
project's promise is that it runs on a bare Python install, and the display
layer is not the place to break that.

The draft loop owns the data; this module only publishes it. `BoardState` is
the handoff, and it is the one piece of shared mutable state in the program, so
it is explicitly locked.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .lineup import bye_conflicts, optimal_lineup, position_counts, remaining_starter_needs

PAGE = Path(__file__).resolve().parent / "web" / "index.html"

POSITION_ORDER = ("QB", "RB", "WR", "TE", "K", "DEF")


class BoardState:
    """Thread-safe latest-value box. Readers never block writers for long."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict = {"status": "starting", "ready": False}

    def set(self, data: dict) -> None:
        with self._lock:
            self._data = data

    def get(self) -> dict:
        with self._lock:
            return self._data


def _player_payload(p, rec=None) -> dict:
    out = {
        "pid": p.pid,
        "name": p.name,
        "position": p.position,
        "team": p.team or "--",
        "bye": p.bye,
        "adp_sleeper": p.adp_sleeper,
        "adp_ffc": p.adp_ffc,
        "projection": round(p.projection, 1),
        "vor": round(p.vor, 1),
        "val": round(p.val, 1),
        "upside": round(p.upside, 1),
        "adp": round(p.adp, 1),
        "injury": p.injury,
        "rookie": bool(p.years_exp is not None and p.years_exp <= 1),
        "backup": bool(p.depth_chart_order and p.depth_chart_order >= 2),
    }
    if rec is not None:
        out["survive"] = round(rec.survive_next, 4)
        out["sim"] = round(rec.sim_score, 1) if rec.sim_score is not None else None
        out["note"] = rec.note
    return out


def serialize(st, recs, outlook, league, diagnostics, meta) -> dict:
    """Flatten everything the board shows into one JSON-ready dict."""
    starters, lineup_points = optimal_lineup(st.my_roster, league)
    needs = remaining_starter_needs(st.my_roster, league)
    conflicts = bye_conflicts(starters, 2)
    counts = position_counts(st.my_roster)
    targets = meta.get("roster_targets") or {}

    started = {q.pid for group in starters.values() for q in group}
    bench = [q for q in st.my_roster if q.pid not in started]

    lineup = []
    for slot in ("QB", "RB", "WR", "TE", "FLEX", "K", "DEF"):
        for q in starters.get(slot, []):
            lineup.append({"slot": slot, **_player_payload(q)})

    rnd = (st.current_pick - 1) // league.teams + 1 if league.teams else 1

    return {
        "ready": True,
        "computing": bool(meta.get("computing")),
        "status": meta.get("status", "?"),
        "current_pick": st.current_pick,
        "total_picks": league.total_picks,
        "round": min(rnd, league.rounds),
        "rounds": league.rounds,
        "teams": league.teams,
        "scoring": league.scoring,
        "my_slot": st.my_slot,
        "on_the_clock": st.on_the_clock,
        "picks_until_mine": st.picks_until_mine,
        "next_picks": st.my_future_picks[:4],
        "recommendations": [_player_payload(r.player, r) for r in recs[: meta.get("top_n", 12)]],
        "outlook": [
            {
                "position": o.position,
                "cost": round(o.cost_of_waiting, 1),
                "best_now": round(o.best_now, 1),
                "expected_next": round(o.expected_next, 1),
                "expected_gone": round(o.expected_gone, 1),
                "best_player": o.best_player.name if o.best_player else None,
            }
            for o in outlook
        ],
        "roster": {
            "lineup": lineup,
            "bench": [_player_payload(q) for q in bench],
            "points": round(lineup_points, 1),
            "size": len(st.my_roster),
            "capacity": league.rounds,
            "needs": {k: v for k, v in needs.items() if v > 0},
            "bye_conflicts": {str(k): v for k, v in sorted(conflicts.items())},
            "depth": [
                {"position": pos, "have": counts.get(pos, 0), "want": targets[pos]}
                for pos in POSITION_ORDER
                if pos in targets
            ],
        },
        "recent": [
            {
                "pick": pk.get("pick_no"),
                "name": (pk.get("metadata") or {}).get("last_name")
                or str(pk.get("player_id", "?")),
            }
            for pk in st.picks[-8:]
        ],
        "meta": {
            "sim_note": meta.get("sim_note", ""),
            "players": diagnostics.get("players_considered", 0),
            "adp_matched": diagnostics.get("adp_matched", 0),
            "adp_rows": diagnostics.get("adp_rows", 0),
        },
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, state: BoardState, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:
        """Silence request logging -- it would bury the terminal output."""

    def _send(self, body: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass  # browser navigated away mid-response

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path == "/state.json":
            payload = json.dumps(self.state.get()).encode("utf-8")
            self._send(payload, "application/json")
            return

        if path in ("/", "/index.html"):
            try:
                body = PAGE.read_bytes()
            except OSError:
                self._send(b"page missing", "text/plain")
                return
            self._send(body, "text/html; charset=utf-8")
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def start_server(state: BoardState, port: int = 8770) -> tuple[ThreadingHTTPServer, str]:
    """Start the board server on a background daemon thread.

    Binds to localhost only -- this exposes your draft, and there is no reason
    for it to be reachable from the rest of the network.
    """
    handler = partial(_Handler, state)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"
