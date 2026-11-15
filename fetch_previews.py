import concurrent.futures
import difflib
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import requests

import audio_cache
from audio_features import extract_vector

SOURCE = os.environ.get("SOURCE_CSV", "data/tracks.csv")
MAX_TRACKS = int(os.environ.get("AUDIO_MAX_TRACKS", "100000"))
BUDGET_MINUTES = float(os.environ.get("AUDIO_MINUTES", "240"))
ITUNES_MINUTES = float(os.environ.get("ITUNES_MINUTES", "20"))
WORKERS = int(os.environ.get("AUDIO_WORKERS", "12"))
PROCS = int(os.environ.get("AUDIO_PROCS", str(os.cpu_count() or 2)))
DEEZER_RPS = float(os.environ.get("DEEZER_RPS", "8"))
ITUNES_RPS = float(os.environ.get("ITUNES_RPS", "0.3"))
RETRY_MISSES = os.environ.get("AUDIO_RETRY_MISS", "0") == "1"
TITLE_MIN = float(os.environ.get("TITLE_MATCH_MIN", "0.85"))
DURATION_TOLERANCE = float(os.environ.get("DURATION_TOLERANCE", "12"))
ABORT_STREAK = int(os.environ.get("ABORT_STREAK", "3"))
USER_AGENT = "preview-feature-fetcher/1.0"


class Transient(Exception):
    pass


class Limiter:
    def __init__(self, rps):
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)

    def penalize(self, seconds):
        with self.lock:
            self.next = max(self.next, time.monotonic() + seconds)


DEEZER = Limiter(DEEZER_RPS)
ITUNES = Limiter(ITUNES_RPS)
_local = threading.local()


def session():
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _local.session = s
    return s


def get_json(url, params, limiter, penalty):
    limiter.wait()
    try:
        response = session().get(url, params=params, timeout=15)
    except requests.RequestException as exc:
        raise Transient(str(exc))
    if response.status_code in (403, 429) or response.status_code >= 500:
        limiter.penalize(penalty)
        raise Transient(f"HTTP {response.status_code}")
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        raise Transient("invalid JSON")


def deezer_search(query):
    data = get_json("https://api.deezer.com/search", {"q": query, "limit": 5}, DEEZER, 10)
    if not isinstance(data, dict):
        return []
    error = data.get("error")
    if error:
        message = str(error.get("message", "")).lower() if isinstance(error, dict) else ""
        code = error.get("code") if isinstance(error, dict) else None
        if code == 4 or "quota" in message:
            DEEZER.penalize(8)
            raise Transient("Deezer quota")
        return []
    found = []
    for item in data.get("data", []):
        url = item.get("preview")
        if url:
            found.append((
                item.get("title", ""),
                (item.get("artist") or {}).get("name", ""),
                item.get("duration"),
                url,
            ))
    return found


def itunes_search(term):
    data = get_json(
        "https://itunes.apple.com/search",
        {"term": term, "media": "music", "entity": "song", "limit": 5},
        ITUNES,
        90,
    )
    if not isinstance(data, dict):
        return []
    found = []
    for item in data.get("results", []):
        url = item.get("previewUrl")
        if url:
            millis = item.get("trackTimeMillis") or 0
            found.append((
                item.get("trackName", ""),
                item.get("artistName", ""),
                millis / 1000.0 if millis else None,
                url,
            ))
    return found


def clean_title(text):
    text = str(text).lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"\s[-\u2013]\s.*$", " ", text)
    text = re.sub(r"\b(feat|ft|featuring)\b.*$", " ", text)
    return re.sub(r"[\W_]+", " ", text).strip()


def clean_artist(text):
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", str(text).lower())
    return re.sub(r"[\W_]+", " ", text).strip()


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def pick(row, candidates):
    _, title, artist, duration = row
    wanted_title = clean_title(title) or clean_artist(title)
    wanted_artist = clean_artist(artist)
    best, best_score = None, 0.0
    for cand_title, cand_artist, cand_seconds, url in candidates:
        got_title = clean_title(cand_title) or clean_artist(cand_title)
        title_score = ratio(wanted_title, got_title)
        if title_score < TITLE_MIN:
            continue
        got_artist = clean_artist(cand_artist)
        artist_ok = bool(wanted_artist and got_artist) and (
            (len(wanted_artist) >= 3 and (wanted_artist in got_artist or got_artist in wanted_artist))
            or ratio(wanted_artist, got_artist) >= 0.75
        )
        if not artist_ok:
            continue
        gap = 0.0
        if duration and cand_seconds:
            gap = abs(cand_seconds - duration)
            if gap > DURATION_TOLERANCE:
                continue
        score = title_score - 0.001 * gap
        if score > best_score:
            best, best_score = url, score
    return best


def quote_free(text):
    return str(text).replace('"', " ").strip()


def find_deezer(row):
    _, title, artist, _ = row
    for query in (
        f'artist:"{quote_free(artist)}" track:"{quote_free(title)}"',
        f"{quote_free(artist)} {quote_free(title)}",
    ):
        url = pick(row, deezer_search(query))
        if url:
            return url
    return None


def find_itunes(row):
    _, title, artist, _ = row
    return pick(row, itunes_search(f"{quote_free(artist)} {quote_free(title)}"))


FINDERS = {"deezer": find_deezer, "itunes": find_itunes}


def download(url):
    try:
        response = session().get(url, timeout=25)
    except requests.RequestException as exc:
        raise Transient(str(exc))
    if response.status_code in (403, 429) or response.status_code >= 500:
        raise Transient(f"HTTP {response.status_code}")
    if response.status_code != 200 or len(response.content) < 30000:
        return None
    return response.content


def handle(pool, provider, row):
    try:
        url = FINDERS[provider](row)
        if url is None:
            return "none", None
        data = download(url)
        if data is None:
            return "none", None
        suffix = ".mp3" if ".mp3" in url.split("?")[0].lower() else ".m4a"
        try:
            vec = pool.submit(extract_vector, data, suffix).result(timeout=240)
        except BrokenProcessPool:
            raise
        except concurrent.futures.TimeoutError:
            return "none", None
        if vec is None:
            return "none", None
        return "ok", vec
    except Transient:
        return "retry", None


def run_phase(label, rows, provider, deadline, pool, results):
    if not rows:
        return True
    miss_status = audio_cache.DEEZER_MISS if provider == "deezer" else audio_cache.MISS
    batch_size = WORKERS * 16
    counts = {"ok": 0, "none": 0, "retry": 0}
    streak = 0
    started = time.monotonic()
    print(f"[{label}] {len(rows)} tracks queued", flush=True)
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as threads:
        for start in range(0, len(rows), batch_size):
            if time.monotonic() >= deadline:
                print(f"[{label}] time budget reached", flush=True)
                return False
            batch = rows[start:start + batch_size]
            futures = {threads.submit(handle, pool, provider, row): row[0] for row in batch}
            batch_retry = 0
            for future in concurrent.futures.as_completed(futures):
                track_id = futures[future]
                outcome, vec = future.result()
                counts[outcome] += 1
                if outcome == "ok":
                    results[track_id] = (audio_cache.DONE, vec)
                elif outcome == "none":
                    results[track_id] = (miss_status, None)
                else:
                    batch_retry += 1
            streak = streak + 1 if batch_retry > 0.9 * len(batch) else 0
            elapsed = max(time.monotonic() - started, 1e-9)
            total = sum(counts.values())
            print(
                f"[{label}] {total}/{len(rows)} ok={counts['ok']} none={counts['none']} "
                f"retry={counts['retry']} rate={total / elapsed:.2f}/s",
                flush=True,
            )
            if streak >= ABORT_STREAK:
                print(f"[{label}] too many failed requests, stopping this phase", flush=True)
                return False
    return True


def flush(results):
    if not results:
        print("No new results to save", flush=True)
        return
    ids = list(results.keys())
    status = [results[i][0] for i in ids]
    feats = np.zeros((len(ids), audio_cache.DIM), dtype=np.float16)
    for position, track_id in enumerate(ids):
        vec = results[track_id][1]
        if vec is not None:
            feats[position] = vec
    path = audio_cache.write_shard(ids, status, feats)
    done = sum(1 for s in status if s == audio_cache.DONE)
    print(f"Saved {len(ids)} results ({done} with features) to {path}", flush=True)


def main():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found, skipping audio previews", flush=True)
        return
    if BUDGET_MINUTES <= 0 or MAX_TRACKS <= 0:
        print("Audio previews disabled", flush=True)
        return

    df = pd.read_csv(
        SOURCE,
        usecols=["track_id", "track_name", "artists", "duration_ms"],
        nrows=MAX_TRACKS,
    ).dropna(subset=["track_id", "track_name", "artists"])
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")

    ids, status, _ = audio_cache.load()
    state = dict(zip(ids.tolist(), status.tolist()))

    rows_all = [
        (
            r.track_id,
            r.track_name,
            str(r.artists).split(";")[0].strip(),
            r.duration_ms / 1000.0 if pd.notna(r.duration_ms) and r.duration_ms > 0 else None,
        )
        for r in df.itertuples(index=False)
    ]

    def is_pending(track_id):
        value = state.get(track_id)
        if value is None:
            return True
        return RETRY_MISSES and value != audio_cache.DONE

    first_pass = [r for r in rows_all if is_pending(r[0]) and state.get(r[0]) != audio_cache.DEEZER_MISS]
    second_pass = [r for r in rows_all if state.get(r[0]) == audio_cache.DEEZER_MISS]
    print(
        f"Cache: {len(state)} known, {sum(1 for v in state.values() if v == audio_cache.DONE)} with features; "
        f"{len(first_pass)} new, {len(second_pass)} awaiting second provider",
        flush=True,
    )
    if not first_pass and not (second_pass and ITUNES_MINUTES > 0):
        print("Nothing to do", flush=True)
        return

    started = time.monotonic()
    deadline = started + BUDGET_MINUTES * 60
    results = {}
    context = multiprocessing.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=PROCS, mp_context=context) as pool:
            finished = run_phase("deezer", first_pass, "deezer", deadline, pool, results)
            if finished and second_pass and ITUNES_MINUTES > 0:
                itunes_deadline = min(deadline, time.monotonic() + ITUNES_MINUTES * 60)
                run_phase("itunes", second_pass, "itunes", itunes_deadline, pool, results)
    except BaseException:
        traceback.print_exc()
    finally:
        flush(results)


if __name__ == "__main__":
    main()
    sys.exit(0)
