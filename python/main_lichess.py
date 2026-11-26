#!/usr/bin/env python3
"""
main_lichess.py -- connects the Talbot UCI engine to Lichess as a BOT account.

  * Accepts / declines incoming challenges by variant, time control, rated/casual,
    bot/human, rating, allow/block lists.
  * Optionally challenges other bots on a timer (online-bot pool filtered by
    rating, or a fixed list).
  * One engine PROCESS per game. Talbot is single-game by design (one board,
    one tree, one batcher), so N concurrent games = N engine processes = N copies
    of the TensorRT model on the GPU.

Talks to the Lichess HTTP API directly with `requests` (no berserk), so clock
values are plain integer milliseconds and moves are plain UCI strings.

Usage:
    export LICHESS_API_TOKEN=lip_xxx          # token needs the bot:play scope
    python main_lichess.py --config play_lichess.yaml
"""

import argparse
import json
import logging
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import chess
import requests
import yaml

ONGOING_STATUSES = {"created", "started"}


# =============================================================================
# Lichess API
# =============================================================================
class LichessAPI:
    def __init__(self, token: str, base_url: str, logger: logging.Logger):
        self.base = base_url.rstrip("/")
        self.log = logger
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": "talbot-lichess-bridge/1.0",
        })
        self._rate_limited_until = 0.0
        self._rate_lock = threading.Lock()

    # ---- plumbing ----------------------------------------------------------
    def _request(self, method, path, retries=3, timeout=15, return_429=False, **kw):
        """Returns the Response (any 2xx/4xx), or None after repeated network/5xx/429 failures.
        return_429=True hands a 429 straight back to the caller instead of pausing everything."""
        url = self.base + path
        for attempt in range(1, retries + 1):
            with self._rate_lock:
                wait = self._rate_limited_until - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.request(method, url, timeout=timeout, **kw)
            except requests.RequestException as e:
                self.log.warning(f"{method} {path} failed ({attempt}/{retries}): {e}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 and return_429:
                return r
            if r.status_code == 429:
                # Lichess asks clients to back off a full minute after a 429.
                self.log.warning(f"429 rate-limited on {method} {path}; pausing all API calls for 60s")
                with self._rate_lock:
                    self._rate_limited_until = time.time() + 60
                continue
            if r.status_code >= 500:
                self.log.warning(f"{method} {path} -> {r.status_code} ({attempt}/{retries})")
                time.sleep(2 ** attempt)
                continue
            return r
        return None

    def open_stream(self, path, params=None):
        r = self.session.get(self.base + path, params=params, stream=True, timeout=(15, 90))
        r.raise_for_status()
        return r

    @staticmethod
    def iter_ndjson(resp):
        """Yields dicts; yields None for keep-alive blank lines (lets callers poll flags)."""
        for raw in resp.iter_lines():
            if not raw:
                yield None
                continue
            try:
                yield json.loads(raw.decode("utf-8"))
            except ValueError:
                continue

    # ---- endpoints ---------------------------------------------------------
    def account(self):
        r = self._request("GET", "/api/account")
        if r is None or r.status_code != 200:
            raise RuntimeError(f"/api/account failed: {None if r is None else (r.status_code, r.text)}")
        return r.json()

    def make_move(self, game_id, uci):
        r = self._request("POST", f"/api/bot/game/{game_id}/move/{uci}")
        if r is None:
            return False, "no response"
        return r.status_code == 200, r.text

    def resign(self, game_id):
        r = self._request("POST", f"/api/bot/game/{game_id}/resign", retries=2)
        return r is not None and r.status_code == 200

    def chat(self, game_id, text, room="player"):
        self._request("POST", f"/api/bot/game/{game_id}/chat", retries=1,
                      data={"room": room, "text": text})

    def accept_challenge(self, challenge_id):
        r = self._request("POST", f"/api/challenge/{challenge_id}/accept")
        return r is not None and r.status_code == 200

    def decline_challenge(self, challenge_id, reason="generic"):
        self._request("POST", f"/api/challenge/{challenge_id}/decline", retries=1,
                      data={"reason": reason})

    def cancel_challenge(self, challenge_id):
        self._request("POST", f"/api/challenge/{challenge_id}/cancel", retries=1)

    def create_challenge(self, username, rated, base_sec, inc_sec, variant):
        """Returns (challenge_id, error, wait_seconds, who_is_limited).
        who_is_limited: "self"     -> our account can't challenge for wait_seconds
                        "opponent" -> that bot hit its daily bot-vs-bot cap; skip it for wait_seconds
                        None       -> ordinary failure"""
        r = self._request("POST", f"/api/challenge/{username}", retries=1, return_429=True, data={
            "rated": "true" if rated else "false",
            "clock.limit": base_sec,
            "clock.increment": inc_sec,
            "variant": variant,
            "color": "random",
        })
        if r is None:
            return None, "no response", None, None
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code == 400 and (body.get("ratelimit") or {}).get("seconds"):
            # e.g. "X played 100 games against other bots today, please wait until ..."
            return None, f"400: {r.text[:300]}", float(body["ratelimit"]["seconds"]), "opponent"
        if r.status_code == 429:
            # Challenge-creation limit. Lichess says how long in ratelimit.seconds.
            secs = None
            try:
                secs = (r.json().get("ratelimit") or {}).get("seconds")
            except ValueError:
                pass
            if secs is None and r.headers.get("Retry-After", "").isdigit():
                secs = int(r.headers["Retry-After"])
            return None, f"429: {r.text[:300]}", float(secs) if secs else 900.0, "self"
        if r.status_code != 200:
            return None, f"{r.status_code}: {r.text[:200]}", None, None
        cid = body.get("id") or (body.get("challenge") or {}).get("id")
        return cid, None, None, None

    def game_result(self, game_id):
        """(status, winner) of a game from the export endpoint; (None, None) on failure."""
        r = self._request("GET", f"/game/export/{game_id}", retries=2,
                          params={"moves": "false", "clocks": "false", "evals": "false"},
                          headers={"Accept": "application/json"})
        if r is None or r.status_code != 200:
            return None, None
        try:
            j = r.json()
        except ValueError:
            return None, None
        return j.get("status"), j.get("winner")

    def online_bots(self, nb=200):
        r = self._request("GET", "/api/bot/online", params={"nb": nb}, timeout=30)
        if r is None or r.status_code != 200:
            return []
        out = []
        for line in r.text.splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        return out


# =============================================================================
# UCI engine wrapper (one process per game)
# =============================================================================
class EngineError(RuntimeError):
    pass


class UciEngine:
    def __init__(self, path, args, options, logger, startup_timeout=120.0):
        self.log = logger
        exe = Path(path).resolve()
        if not exe.exists():
            raise EngineError(f"engine not found: {exe}")
        self.log.info(f"Launching engine: {exe} {' '.join(args)}")
        self.proc = subprocess.Popen(
            [str(exe), *args],
            cwd=str(exe.parent),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

        # Talbot loads the TensorRT engine BEFORE reading stdin, so uciok
        # arriving means the model is loaded.
        self.send("uci")
        self._wait_for("uciok", startup_timeout)
        for name, value in options.items():
            if value is None or value == "":
                continue
            self.send(f"setoption name {name} value {value}")
        self.send("isready")            # tb_init (SyzygyPath) runs before this returns
        self._wait_for("readyok", startup_timeout)
        self.send("ucinewgame")

    def _pump_stdout(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def _pump_stderr(self):
        for line in self.proc.stderr:
            line = line.rstrip()
            # Only real errors surface; the rest is dropped.
            if any(k in line for k in ("Fatal", "ERROR", "Error", "error", "terminate")):
                self.log.warning(f"[engine stderr] {line}")
            else:
                self.log.debug(f"[engine stderr] {line}")

    def send(self, cmd):
        self.log.debug(f">> {cmd}")
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngineError(f"engine pipe closed: {e}")

    def _wait_for(self, prefix, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError(f"timed out waiting for '{prefix}'")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise EngineError(f"engine exited (code {self.proc.poll()}) while waiting for '{prefix}'")
            self.log.debug(f"<< {line}")
            if line.startswith(prefix):
                return line

    def search(self, initial_fen, moves, wtime, btime, winc, binc, abort_event, my_time_ms,
               movetime_ms=None):
        """Returns (bestmove_uci_or_None, last_info_line).
        movetime_ms set -> fixed-time search (`go movetime`) instead of clock-based."""
        # Drop anything stale (shouldn't exist: we always consume up to bestmove).
        while not self.lines.empty():
            if self.lines.get_nowait() is None:
                raise EngineError("engine exited")

        pos = "position startpos" if initial_fen is None else f"position fen {initial_fen}"
        if moves:
            pos += " moves " + " ".join(moves)
        self.send(pos)
        if movetime_ms is not None:
            self.send(f"go movetime {movetime_ms}")
            budget_ms = min(movetime_ms, my_time_ms)
        else:
            self.send(f"go wtime {max(1, wtime)} btime {max(1, btime)} winc {winc} binc {binc}")
            budget_ms = my_time_ms

        # Watchdog: if the engine overruns its budget + 2s, force a stop;
        # if it still doesn't answer 10s after that, give up on it.
        stop_at = time.monotonic() + budget_ms / 1000.0 + 2.0
        give_up_at = stop_at + 10.0
        stop_sent = False
        last_info = ""
        while True:
            try:
                line = self.lines.get(timeout=0.05)
            except queue.Empty:
                now = time.monotonic()
                if not stop_sent and (abort_event.is_set() or now > stop_at):
                    self.send("stop")
                    stop_sent = True
                if now > give_up_at:
                    raise EngineError("engine did not return bestmove after stop")
                continue
            if line is None:
                raise EngineError(f"engine exited mid-search (code {self.proc.poll()})")
            if line.startswith("info"):
                if " score " in line:
                    last_info = line
                self.log.debug(f"<< {line}")
                continue
            if line.startswith("bestmove"):
                self.log.debug(f"<< {line}")
                parts = line.split()
                mv = parts[1] if len(parts) > 1 else None
                return (None if mv in (None, "0000", "(none)") else mv), last_info
            self.log.debug(f"<< {line}")

    def quit(self, timeout=15.0):
        if self.proc.poll() is not None:
            return
        try:
            self.send("stop")
            self.send("quit")       # Talbot writes game.pgn on quit (if log dir is set)
        except EngineError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log.warning("Engine did not exit after quit; killing.")
            self.proc.kill()
            self.proc.wait()


def parse_info(info_line):
    """Pulls depth / nodes / nps / time out of a UCI info line. Missing fields are absent."""
    t = info_line.split()
    out = {}
    for key in ("depth", "nodes", "nps", "time"):
        if key in t:
            i = t.index(key)
            if i + 1 < len(t) and t[i + 1].isdigit():
                out[key] = int(t[i + 1])
    return out


def parse_score(info_line):
    t = info_line.split()
    for i, tok in enumerate(t):
        if tok == "score" and i + 2 < len(t):
            return f"{t[i + 1]} {t[i + 2]}"
    return "?"


# =============================================================================
# One game
# =============================================================================
class Game(threading.Thread):
    def __init__(self, bot, game_id, slot, start_info):
        super().__init__(name=f"game-{game_id}", daemon=True)
        self.bot = bot
        self.api = bot.api
        self.game_id = game_id
        self.slot = slot
        self.over = threading.Event()
        self.events = queue.Queue()
        self.engine = None
        self.resp = None

        opp = start_info.get("opponent") or {}
        self.opponent = opp.get("username") or opp.get("id") or "unknown"
        self.color = chess.WHITE if start_info.get("color") == "white" else chess.BLACK
        self.initial_fen = None          # None = standard start position
        self.last_searched_ply = -1
        self.finished = False
        self.search_stats = []           # parse_info() of the final info line of each of our searches

        self.log = bot.make_game_logger(game_id, self.opponent)

    # ---- stream reader (own thread) -----------------------------------------
    def _reader(self):
        failures = 0
        while not self.over.is_set() and not self.bot.stop.is_set():
            try:
                self.resp = self.api.open_stream(f"/api/bot/game/stream/{self.game_id}")
                failures = 0
                for ev in self.api.iter_ndjson(self.resp):
                    # Keep reading even if `over` is set (gameFinish on the main stream
                    # often beats the final gameState); Lichess closes this stream at game end.
                    if self.bot.stop.is_set():
                        break
                    if ev is None:
                        continue
                    ev["_recv"] = time.monotonic()
                    self.events.put(ev)
                    st = ev.get("state") if ev.get("type") == "gameFull" else (
                        ev if ev.get("type") == "gameState" else None)
                    if st and st.get("status") not in ONGOING_STATUSES:
                        self.over.set()     # lets an in-flight search be stopped
            except Exception as e:
                if self.over.is_set() or self.bot.stop.is_set():
                    break
                failures += 1
                self.log.warning(f"Game stream error ({failures}): {e}")
                if failures >= 5:
                    self.log.error("Game stream failed repeatedly; giving up on this game.")
                    self.over.set()
                    break
                time.sleep(2 * failures)
                continue
            if not self.over.is_set():
                # Stream closed without a terminal status: reconnect (Lichess
                # re-sends gameFull on reconnect).
                self.log.info("Game stream ended without a result; reconnecting.")
                time.sleep(1)
        self.events.put(None)

    # ---- main --------------------------------------------------------------
    def run(self):
        try:
            threading.Thread(target=self._reader, daemon=True, name=f"stream-{self.game_id}").start()
            try:
                self.engine = self.bot.launch_engine(self.slot, self.opponent, self.log)
            except Exception as e:
                self.log.error(f"Engine failed to start: {e}. Resigning.")
                self.bot.log.error(f"[{self.game_id}] engine failed to start: {e}")
                self.api.resign(self.game_id)
                return
            greeting = self.bot.cfg["games"].get("greeting")
            if greeting:
                self.api.chat(self.game_id, greeting)
            self._loop()
        except Exception as e:
            self.log.exception(f"Game crashed: {e}")
            self.bot.log.error(f"[{self.game_id}] game thread crashed: {e}")
            if not self.over.is_set():
                self.api.resign(self.game_id)
        finally:
            self.over.set()
            if not self.finished:
                # Final state never reached us (stream dropped, shutdown, engine failure).
                # Ask Lichess for the result so the game still gets its line in main.log.
                status, winner = self.api.game_result(self.game_id)
                self._report_result(status or "unknown", winner)
            if self.resp is not None:
                try:
                    self.resp.close()
                except Exception:
                    pass
            if self.engine is not None:
                self.engine.quit()
            self.bot.on_game_end(self)

    def _loop(self):
        while not self.finished:
            try:
                ev = self.events.get(timeout=1.0)
            except queue.Empty:
                if self.bot.stop.is_set():
                    return
                continue
            if ev is None:
                return
            t = ev.get("type")
            if t == "gameFull":
                self._on_full(ev)
                st = dict(ev.get("state") or {})
                st["_recv"] = ev["_recv"]
                self._on_state(st)
            elif t == "gameState":
                self._on_state(ev)
            elif t == "chatLine":
                self.log.info(f"chat [{ev.get('room')}] {ev.get('username')}: {ev.get('text')}")
            elif t == "opponentGone":
                self.log.info(f"opponentGone: {ev}")

    def _report_result(self, status, winner):
        if self.finished:
            return
        self.finished = True
        if winner:
            result = "WON" if (winner == "white") == (self.color == chess.WHITE) else "LOST"
        else:
            result = "DRAW/ABORT"
        self.bot.log.info(f"[{self.game_id}] vs {self.opponent}: {status}, winner={winner} ({result})"
                          f"{self._stats_summary()}")

    def _stats_summary(self):
        st = self.search_stats
        if not st:
            return ""
        nodes = sum(x["nodes"] for x in st)
        ms = sum(x.get("time", 0) for x in st)
        # Time-weighted NPS (total nodes / total search time), not a mean of per-move NPS.
        nps = int(nodes * 1000 / ms) if ms > 0 else 0
        depths = [x["depth"] for x in st if "depth" in x]
        avg_depth = sum(depths) / len(depths) if depths else 0
        return (f" | {len(st)} moves searched | avg nps {nps} | avg nodes {nodes // len(st)}"
                f" | avg depth {avg_depth:.1f}")

    def _on_full(self, ev):
        fen = ev.get("initialFen") or "startpos"
        self.initial_fen = None if fen == "startpos" else fen
        white = (ev.get("white") or {}).get("id", "")
        self.color = chess.WHITE if white.lower() == self.bot.bot_id else chess.BLACK
        clock = ev.get("clock") or {}
        self.log.info(
            f"gameFull: {ev.get('white', {}).get('name')} vs {ev.get('black', {}).get('name')} | "
            f"variant={ev.get('variant', {}).get('key')} rated={ev.get('rated')} "
            f"clock={clock.get('initial')}ms+{clock.get('increment')}ms | "
            f"we are {'white' if self.color == chess.WHITE else 'black'} | initialFen={fen}")

    def _board(self, moves):
        b = chess.Board(self.initial_fen) if self.initial_fen else chess.Board()
        for m in moves:
            b.push_uci(m)
        return b

    def _on_state(self, st):
        status = st.get("status")
        if status not in ONGOING_STATUSES:
            self._report_result(status, st.get("winner"))
            return

        moves = (st.get("moves") or "").split()
        try:
            board = self._board(moves)
        except ValueError as e:
            self.log.error(f"Could not replay moves {moves}: {e}")
            return

        if board.turn != self.color or board.is_game_over():
            return
        if len(moves) == self.last_searched_ply:
            return          # already answered this position
        self.last_searched_ply = len(moves)

        # Lichess clock values are as of when the event was sent. Deduct the
        # time this event sat in our queue. Network latency is the engine's
        # MoveOverhead's job.
        lag_ms = int((time.monotonic() - st.get("_recv", time.monotonic())) * 1000)
        wtime, btime = int(st.get("wtime", 0)), int(st.get("btime", 0))
        winc, binc = int(st.get("winc", 0)), int(st.get("binc", 0))
        if self.color == chess.WHITE:
            wtime = max(1, wtime - lag_ms)
            my_time = wtime
        else:
            btime = max(1, btime - lag_ms)
            my_time = btime

        # Opening: short fixed think, paid for by the increment, so the clock
        # grows through the opening instead of being spent on it.
        movetime = None
        op = self.bot.cfg.get("opening") or {}
        fast_plies = int(op.get("fast_plies", 0) or 0)
        if fast_plies and len(moves) < fast_plies:
            my_inc = winc if self.color == chess.WHITE else binc
            mt = int(float(op.get("increment_fraction", 0.6)) * my_inc)
            mt = max(int(op.get("min_ms", 300)), min(mt, int(op.get("max_ms", 2000))))
            movetime = max(50, min(mt, my_time // 20))   # never more than 5% of our clock

        t0 = time.monotonic()
        best, info = self.engine.search(self.initial_fen, moves, wtime, btime, winc, binc,
                                        self.over, my_time, movetime_ms=movetime)
        think_ms = int((time.monotonic() - t0) * 1000)
        st_info = parse_info(info) if info else {}
        if "nodes" in st_info:           # tablebase moves report no nodes; skip them
            self.search_stats.append(st_info)

        if self.over.is_set():
            return
        if best is None:
            self.log.error("Engine returned no move. Resigning.")
            self.api.resign(self.game_id)
            return
        try:
            legal = chess.Move.from_uci(best) in board.legal_moves
        except ValueError:
            legal = False
        if not legal:
            self.log.critical(f"Engine returned illegal move {best} in {board.fen()}. Resigning.")
            self.api.resign(self.game_id)
            return

        ok, text = self.api.make_move(self.game_id, best)
        san = board.san(chess.Move.from_uci(best))
        if ok:
            self.log.info(f"ply {len(moves) + 1}: {san} ({best}) | {parse_score(info)} | "
                          f"think {think_ms}ms{' (opening)' if movetime is not None else ''} | clock {my_time}ms")
        else:
            # Usually the game ended while we were thinking.
            self.log.warning(f"Move {best} rejected: {text.strip()[:200]}")


# =============================================================================
# Bot (event stream, challenge filter, matchmaking)
# =============================================================================
def perf_for(base_sec, inc_sec):
    est = base_sec + 40 * inc_sec
    if est < 30:
        return "ultraBullet"
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


class TalbotLichessBot:
    def __init__(self, cfg, config_path):
        self.cfg = cfg
        self.config_dir = Path(config_path).resolve().parent
        self.stop = threading.Event()
        self.lock = threading.RLock()

        self.games = {}                 # game_id -> Game
        self.reserved = {}              # challenge/game id -> monotonic time (accepted or sent, no gameStart yet)
        self.pending_out = None         # (challenge_id, opponent, sent_at)
        self.cooldown = {}              # opponent id (lower) -> monotonic time when allowed again
        self.games_finished = 0
        self.max_games = int(cfg["games"]["max_concurrent"])
        self.free_slots = list(range(self.max_games))
        self._bot_cache = (0.0, [])

        log_cfg = cfg["logging"]
        # Empty / null dir = no log files and no folders created (console only).
        self.log_dir = None
        if log_cfg.get("dir"):
            base = Path(log_cfg["dir"])
            if not base.is_absolute():
                base = self.config_dir / base
            self.log_dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = getattr(logging, str(log_cfg.get("level", "INFO")).upper())
        self.engine_logging = bool(log_cfg.get("engine_logging", False))

        self.log = logging.getLogger("talbot.main")
        self.log.setLevel(self.log_level)
        self.log.propagate = False
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        if self.log_dir is not None:
            fh = logging.FileHandler(self.log_dir / "main.log")
            fh.setFormatter(fmt)
            self.log.addHandler(fh)
        if log_cfg.get("console", True) or self.log_dir is None:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            self.log.addHandler(sh)

        token = cfg["lichess"].get("token") or os.getenv("LICHESS_API_TOKEN")
        if not token:
            raise SystemExit("No Lichess token: set lichess.token or LICHESS_API_TOKEN.")
        self.api = LichessAPI(token, cfg["lichess"].get("base_url", "https://lichess.org"), self.log)
        self.bot_id = None

    # ---- helpers -----------------------------------------------------------
    def make_game_logger(self, game_id, opponent):
        # No per-game files. A game's warnings/errors go to main.log / console.
        lg = logging.getLogger(f"talbot.game.{game_id}")
        lg.propagate = False
        if not lg.handlers:
            lg.setLevel(logging.WARNING)
            for h in self.log.handlers:
                lg.addHandler(h)
        return lg

    def launch_engine(self, slot, opponent, game_log):
        ecfg = self.cfg["engine"]
        path = Path(ecfg["path"])
        if not path.is_absolute():
            path = self.config_dir / path
        args = ["--opponent", opponent]
        # Engine logs (uci_main.log, inference_batcher.log, game.pgn) only when
        # engine_logging is on. Talbot makes its own <timestamp>_vs-<opp>_pid<N>/ subfolder.
        if self.log_dir is not None and self.engine_logging:
            args += ["--log-dir", str(self.log_dir)]
        slot_cores = ecfg.get("slot_cores") or []
        if slot < len(slot_cores) and slot_cores[slot]:
            sc = slot_cores[slot]
            if sc.get("worker"):
                args += ["--worker-cores", str(sc["worker"])]
            if sc.get("batcher"):
                args += ["--batcher-cores", str(sc["batcher"])]
        args += [str(a) for a in (ecfg.get("extra_args") or [])]
        # Opening-only policy softmax temperature (engine UCI options), applied
        # over the same plies as opening.fast_plies. null policy_temp = don't
        # send either option -> play_uci.yaml value throughout.
        op = self.cfg.get("opening") or {}
        policy_temp = op.get("policy_temp")
        options = {
            "MoveOverhead": ecfg.get("move_overhead_ms"),
            "SyzygyPath": ecfg.get("syzygy_path"),
            "OpeningPolicyTemp": policy_temp,
            "OpeningPolicyPlies": int(op.get("fast_plies", 0) or 0) if policy_temp is not None else None,
        }
        return UciEngine(path, args, options, game_log,
                         startup_timeout=float(ecfg.get("startup_timeout_sec", 120)))

    def _expire_reservations(self):
        now = time.monotonic()
        with self.lock:
            for k in [k for k, t in self.reserved.items() if now - t > 90]:
                del self.reserved[k]

    def _busy_count(self):
        with self.lock:
            return len(self.games) + len(self.reserved)

    def _limit_reached(self):
        total = self.cfg["games"].get("total_games")
        if total is None:
            return False
        with self.lock:
            return self.games_finished + len(self.games) + len(self.reserved) >= int(total)

    # ---- lifecycle ---------------------------------------------------------
    def run(self):
        acct = self.api.account()
        if acct.get("title") != "BOT":
            raise SystemExit(f"Account '{acct.get('username')}' is not a BOT account "
                             "(upgrade it via POST /api/bot/account/upgrade first).")
        self.bot_id = acct["id"].lower()
        self.log.info(f"Logged in as {acct.get('username')} (BOT). Logs: {self.log_dir or 'console only'}")

        if self.cfg["matchmaking"].get("enabled"):
            threading.Thread(target=self._matchmaking_loop, daemon=True, name="matchmaking").start()

        try:
            self._event_loop()
        except KeyboardInterrupt:
            self.log.info("Interrupted.")
        finally:
            self.shutdown()

    def _event_loop(self):
        backoff = 1
        while not self.stop.is_set():
            try:
                resp = self.api.open_stream("/api/stream/event")
                self.log.info("Connected to event stream.")
                backoff = 1
                for ev in self.api.iter_ndjson(resp):
                    if self.stop.is_set():
                        break
                    if ev is not None:
                        try:
                            self._handle_event(ev)
                        except Exception as e:
                            self.log.exception(f"Error handling event {ev.get('type')}: {e}")
                    if self._should_exit():   # checked on keep-alives too
                        self.log.info("total_games reached and all games finished. Exiting.")
                        return
            except requests.RequestException as e:
                self.log.warning(f"Event stream error: {e}. Reconnecting in {backoff}s.")
            if self.stop.is_set() or self._should_exit():
                return
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _should_exit(self):
        total = self.cfg["games"].get("total_games")
        if total is None or not self.cfg["games"].get("exit_when_done", True):
            return False
        with self.lock:
            return self.games_finished >= int(total) and not self.games and not self.reserved

    def shutdown(self):
        if self.stop.is_set() and not self.games:
            return
        self.stop.set()
        self.log.info("Shutting down...")
        with self.lock:
            pending = self.pending_out
            games = list(self.games.values())
        if pending:
            self.api.cancel_challenge(pending[0])
        for g in games:
            if self.cfg["games"].get("resign_on_shutdown", True) and not g.finished:
                self.log.info(f"Resigning {g.game_id}")
                self.api.resign(g.game_id)
            g.over.set()
            if g.resp is not None:
                try:
                    g.resp.close()
                except Exception:
                    pass
        for g in games:
            g.join(timeout=20)
            if g.is_alive() and g.engine is not None:
                g.engine.proc.kill()
        self.log.info("Shutdown complete.")

    # ---- events ------------------------------------------------------------
    def _handle_event(self, ev):
        t = ev.get("type")
        self.log.debug(f"event: {ev}")
        if t == "challenge":
            self._on_challenge(ev["challenge"])
        elif t == "gameStart":
            self._on_game_start(ev["game"])
        elif t == "gameFinish":
            g = ev.get("game", {})
            gid = g.get("gameId") or g.get("id")
            with self.lock:
                game = self.games.get(gid)
            if game:
                game.over.set()
        elif t in ("challengeDeclined", "challengeCanceled"):
            ch = ev.get("challenge", {})
            cid = ch.get("id")
            with self.lock:
                self.reserved.pop(cid, None)
                if self.pending_out and self.pending_out[0] == cid:
                    opp = self.pending_out[1]
                    self.pending_out = None
                    if t == "challengeDeclined":
                        self._set_cooldown(opp)
                        self.log.info(f"{opp} declined our challenge "
                                      f"({ch.get('declineReasonKey') or ch.get('declineReason')}).")

    def _set_cooldown(self, opponent):
        mins = float(self.cfg["matchmaking"].get("decline_cooldown_min", 60))
        self.cooldown[opponent.lower()] = time.monotonic() + mins * 60

    def _on_game_start(self, g):
        gid = g.get("gameId") or g.get("id")
        with self.lock:
            self.reserved.pop(gid, None)
            if self.pending_out and self.pending_out[0] == gid:
                self.pending_out = None
            if gid in self.games:
                return
            if not self.free_slots:
                self.log.warning(f"gameStart {gid} but all {self.max_games} slot(s) busy. Resigning it.")
                full = True
            else:
                full = False
                slot = self.free_slots.pop(0)
                game = Game(self, gid, slot, g)
                self.games[gid] = game
        if full:
            self.api.resign(gid)
            return
        opp = (g.get("opponent") or {})
        self.log.info(f"Game {gid} started vs {opp.get('username')} ({opp.get('rating')}), "
                      f"{g.get('speed')}, slot {slot}. Active: {len(self.games)}")
        game.start()

    def on_game_end(self, game):
        with self.lock:
            if self.games.pop(game.game_id, None) is not None:
                self.free_slots.append(game.slot)
                self.free_slots.sort()
                self.games_finished += 1
            n = len(self.games)
        self.log.info(f"Game {game.game_id} cleaned up. Active: {n}. Finished this session: {self.games_finished}")

    # ---- incoming challenges -----------------------------------------------
    def _evaluate_challenge(self, ch):
        """Returns None to accept, or a Lichess decline-reason key."""
        a = self.cfg["challenge_accept"]
        challenger = ch.get("challenger") or {}
        cid = (challenger.get("id") or "").lower()

        if not a.get("enabled", True) or self.stop.is_set():
            return "later"
        if self._limit_reached():
            return "later"
        self._expire_reservations()
        if self._busy_count() >= self.max_games:
            return "later"

        if ch.get("variant", {}).get("key") not in a.get("variants", ["standard"]):
            return "variant"

        tc = ch.get("timeControl") or {}
        if tc.get("type") != "clock":
            return "timeControl"
        base, inc = int(tc.get("limit", 0)), int(tc.get("increment", 0))
        if base < a["min_base_sec"] or inc < a["min_increment_sec"]:
            return "tooFast"
        if base > a["max_base_sec"] or inc > a["max_increment_sec"]:
            return "tooSlow"

        if ch.get("rated") and not a.get("accept_rated", True):
            return "casual"
        if not ch.get("rated") and not a.get("accept_casual", True):
            return "rated"

        is_bot = challenger.get("title") == "BOT"
        if is_bot and not a.get("accept_bots", True):
            return "noBot"
        if not is_bot and not a.get("accept_humans", True):
            return "onlyBot"

        allow = [x.lower() for x in (a.get("allowlist") or [])]
        block = [x.lower() for x in (a.get("blocklist") or [])]
        if cid in block or (allow and cid not in allow):
            return "generic"

        rating = challenger.get("rating")
        if rating is not None:
            if a.get("min_rating") is not None and rating < a["min_rating"]:
                return "generic"
            if a.get("max_rating") is not None and rating > a["max_rating"]:
                return "generic"
        return None

    def _on_challenge(self, ch):
        challenger = (ch.get("challenger") or {})
        if (challenger.get("id") or "").lower() == self.bot_id:
            return                          # our own outgoing challenge echoed back
        cid = ch["id"]
        tc = ch.get("timeControl") or {}
        desc = (f"{cid} from {challenger.get('name') or challenger.get('id')} "
                f"({challenger.get('rating')}, {'BOT' if challenger.get('title') == 'BOT' else 'human'}) "
                f"{ch.get('variant', {}).get('key')} {tc.get('show', tc.get('type'))} "
                f"{'rated' if ch.get('rated') else 'casual'}")
        reason = self._evaluate_challenge(ch)
        if reason is None:
            with self.lock:
                self.reserved[cid] = time.monotonic()   # game id == challenge id
            if self.api.accept_challenge(cid):
                self.log.info(f"Accepted challenge {desc}")
            else:
                with self.lock:
                    self.reserved.pop(cid, None)
                self.log.warning(f"Accept failed for {desc}")
        else:
            self.log.info(f"Declined challenge {desc}: {reason}")
            self.api.decline_challenge(cid, reason)

    # ---- outgoing challenges -----------------------------------------------
    def _pick_opponent(self, perf):
        m = self.cfg["matchmaking"]
        now = time.monotonic()
        block = {x.lower() for x in (m.get("blocklist") or [])}

        def ok(name):
            n = name.lower()
            return n != self.bot_id and n not in block and self.cooldown.get(n, 0) <= now

        if m.get("opponent_source", "online") == "list":
            cands = [o for o in (m.get("opponents") or []) if ok(o)]
            return random.choice(cands) if cands else None

        fetched_at, bots = self._bot_cache
        if now - fetched_at > 600 or not bots:
            bots = self.api.online_bots(int(m.get("online_pool_size", 200)))
            self._bot_cache = (now, bots)
            self.log.info(f"Fetched {len(bots)} online bots.")
        lo, hi = m.get("min_rating"), m.get("max_rating")
        cands = []
        for b in bots:
            if not ok(b.get("id", "")) or b.get("disabled") or b.get("tosViolation"):
                continue
            p = (b.get("perfs") or {}).get(perf) or {}
            r = p.get("rating")
            if r is None or p.get("prov"):
                continue
            if (lo is not None and r < lo) or (hi is not None and r > hi):
                continue
            cands.append((b.get("username") or b["id"], r))
        if not cands:
            return None
        name, r = random.choice(cands)
        self.log.debug(f"Picked {name} ({perf} {r}) from {len(cands)} candidates.")
        return name

    def _matchmaking_loop(self):
        m = self.cfg["matchmaking"]
        interval = float(m.get("interval_sec", 30))
        timeout = float(m.get("challenge_timeout_sec", 60))
        reserve = int(m.get("reserve_slots_for_incoming", 0))
        self.log.info("Matchmaking thread started.")
        paused_until = 0.0
        while not self.stop.wait(interval):
            try:
                self._expire_reservations()
                if time.monotonic() < paused_until:
                    continue
                with self.lock:
                    pending = self.pending_out
                if pending:
                    cid, opp, sent = pending
                    if time.monotonic() - sent > timeout:
                        self.log.info(f"Challenge {cid} to {opp} not answered in {timeout:.0f}s; cancelling.")
                        self.api.cancel_challenge(cid)
                        with self.lock:
                            self.pending_out = None
                            self.reserved.pop(cid, None)
                        self._set_cooldown(opp)
                    continue
                if self._limit_reached():
                    continue
                if self._busy_count() >= self.max_games - reserve:
                    continue

                base, inc = random.choice(m["time_controls"])
                perf = perf_for(base, inc)
                opp = self._pick_opponent(perf)
                if not opp:
                    self.log.info(f"No eligible opponent for {perf} {base}+{inc}.")
                    continue
                cid, err, limit_secs, limited = self.api.create_challenge(opp, bool(m.get("rated", True)), int(base), int(inc),
                                                     m.get("variant", "standard"))
                if cid:
                    with self.lock:
                        self.pending_out = (cid, opp, time.monotonic())
                        self.reserved[cid] = time.monotonic()
                    self.log.info(f"Challenged {opp}: {base}s+{inc}s "
                                  f"{'rated' if m.get('rated', True) else 'casual'} ({cid})")
                elif limited == "opponent":
                    self.cooldown[opp.lower()] = time.monotonic() + limit_secs
                    self.log.info(f"{opp} is at its daily bot-vs-bot limit; skipping it for "
                                  f"{limit_secs / 3600:.1f} h.")
                elif limited == "self":
                    # Our account hit Lichess's challenge limit -- not the opponent's fault.
                    # Only matchmaking pauses; moves and accepting challenges carry on.
                    paused_until = time.monotonic() + limit_secs
                    resume = datetime.now().timestamp() + limit_secs
                    self.log.warning(f"Lichess challenge limit hit ({err}). Pausing matchmaking for "
                                     f"{limit_secs / 60:.0f} min (until "
                                     f"{datetime.fromtimestamp(resume).strftime('%H:%M')}).")
                else:
                    self.log.warning(f"Challenge to {opp} failed: {err}")
                    self._set_cooldown(opp)
            except Exception as e:
                self.log.exception(f"Matchmaking error: {e}")
        self.log.info("Matchmaking thread stopped.")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Run the Talbot UCI engine as a Lichess bot.")
    ap.add_argument("--config", default="/app/config/play_lichess.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bot = TalbotLichessBot(cfg, args.config)

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    bot.run()


if __name__ == "__main__":
    main()