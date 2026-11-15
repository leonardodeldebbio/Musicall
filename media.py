import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TRACKS = "data/tracks.csv"
CACHE = "cache/media.csv"
CACHE_COLS = ["track_id", "itunes_album", "cover_url", "preview_url", "checked"]
URL = "https://itunes.apple.com/search"

MINUTES = float(os.environ.get("MEDIA_MINUTES", "20"))
DELAY = float(os.environ.get("MEDIA_DELAY", "3"))
RETRY_DAYS = int(os.environ.get("MEDIA_RETRY_DAYS", "60"))
REQUIRE_MEDIA = os.environ.get("REQUIRE_MEDIA", "0") == "1"


class RateLimited(Exception):
    pass


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[\(\[].*?[\)\]]", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def lookup(session, artists, track):
    first_artist = re.split(r"[;,]", str(artists))[0].strip()
    t, a = norm(track), norm(first_artist)
    if not t or not a:
        return None
    r = session.get(
        URL,
        params={"term": f"{first_artist} {track}", "media": "music",
                "entity": "song", "limit": 10},
        timeout=20,
    )
    if r.status_code in (403, 429):
        raise RateLimited()
    r.raise_for_status()
    for item in r.json().get("results", []):
        if not item.get("previewUrl"):
            continue
        if norm(item.get("trackName", "")) == t and a in norm(item.get("artistName", "")):
            cover = item.get("artworkUrl100", "").replace("100x100bb", "600x600bb")
            return {
                "itunes_album": item.get("collectionName", ""),
                "cover_url": cover,
                "preview_url": item["previewUrl"],
            }
    return None


def save_cache(cache):
    pd.DataFrame(list(cache.values()), columns=CACHE_COLS).to_csv(CACHE, index=False)


def main():
    df = pd.read_csv(TRACKS, low_memory=False)

    cache = {}
    if os.path.exists(CACHE):
        old = pd.read_csv(CACHE, dtype=str).fillna("")
        cache = {r["track_id"]: r for r in old.to_dict("records")}
    print(f"Cache entries: {len(cache)}")

    now = datetime.now(timezone.utc)
    limit_date = (now - timedelta(days=RETRY_DAYS)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")

    pending = []
    for row in df[["track_id", "artists", "track_name"]].itertuples(index=False):
        rec = cache.get(row.track_id)
        if rec is None or (not rec["preview_url"] and rec["checked"] < limit_date):
            pending.append(row)
    print(f"Tracks to look up: {len(pending)}")

    deadline = time.time() + MINUTES * 60
    session = requests.Session()
    i = done = fails = 0
    while i < len(pending) and time.time() < deadline and fails < 5:
        row = pending[i]
        try:
            found = lookup(session, row.artists, row.track_name)
        except RateLimited:
            fails += 1
            print("Rate limited, waiting 60s")
            time.sleep(60)
            continue
        except requests.RequestException as exc:
            print(f"Request error: {exc}")
            found = None
        fails = 0
        cache[row.track_id] = {
            "track_id": row.track_id,
            "itunes_album": (found or {}).get("itunes_album", ""),
            "cover_url": (found or {}).get("cover_url", ""),
            "preview_url": (found or {}).get("preview_url", ""),
            "checked": today,
        }
        i += 1
        done += 1
        if done % 200 == 0:
            save_cache(cache)
            print(f"Looked up {done}/{len(pending)}")
        time.sleep(DELAY)

    save_cache(cache)
    print(f"Lookups this run: {done}")

    media = pd.DataFrame(list(cache.values()), columns=CACHE_COLS)
    df = df.merge(media[["track_id", "itunes_album", "cover_url", "preview_url"]],
                  on="track_id", how="left")
    for c in ["album_name", "itunes_album", "cover_url", "preview_url"]:
        df[c] = df[c].fillna("").astype(str)
    df["album_name"] = df["album_name"].where(df["album_name"].str.strip() != "", df["itunes_album"])
    df = df.drop(columns="itunes_album")

    if REQUIRE_MEDIA:
        df = df[df["preview_url"] != ""]
    print(f"Tracks with preview: {(df['preview_url'] != '').sum()} / {len(df)}")

    df.to_csv(TRACKS, index=False)


if __name__ == "__main__":
    main()
