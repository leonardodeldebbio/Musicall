import json
import os
import sys

import numpy as np
import pandas as pd

SOURCE = os.environ.get("SOURCE_CSV", "data/tracks.csv")
OUT_DIR = os.environ.get("OUT_DIR", "docs/data")
K = int(os.environ.get("NEIGHBORS", "20"))
GENRE_WEIGHT = float(os.environ.get("GENRE_WEIGHT", "2.0"))
CHUNK = int(os.environ.get("CHUNK", "1024"))
MIN_ARTIST_TRACKS = int(os.environ.get("MIN_ARTIST_TRACKS", "1"))

NUMERIC = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "explicit", "mode", "time_signature",
]


def top_k(matrix, k, chunk=CHUNK):
    x = matrix.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms == 0, 1, norms)
    n = len(x)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = x[start:end] @ x.T
        block[np.arange(end - start), np.arange(start, end)] = -2
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[start:end] = np.take_along_axis(part, order, axis=1)
        sim[start:end] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def feature_blocks(df):
    cols = [c for c in NUMERIC if c in df.columns]
    num = df[cols].astype("float64")
    if "duration_ms" in num:
        num["duration_ms"] = np.log1p(num["duration_ms"])
    num = num.fillna(num.median())
    std = num.std().replace(0, 1)
    z = ((num - num.mean()) / std).clip(-4, 4).to_numpy(np.float32)
    parts = [z]
    if "key" in df.columns:
        angle = 2 * np.pi * df["key"].fillna(0).to_numpy(np.float32) / 12
        parts.append(0.5 * np.stack([np.sin(angle), np.cos(angle)], axis=1))
    base = np.hstack(parts).astype(np.float32)
    if "track_genre" in df.columns:
        genres = pd.get_dummies(df["track_genre"].fillna("unknown")).astype(np.float32)
        genre_names = list(genres.columns)
        genre = genres.to_numpy()
    else:
        genre_names = ["unknown"]
        genre = np.ones((len(df), 1), dtype=np.float32)
    return base, genre, genre_names


def write_kind(kind, catalog, idx, sim):
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog["n"] = int(idx.shape[0])
    catalog["k"] = int(idx.shape[1])
    with open(f"{OUT_DIR}/{kind}.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, separators=(",", ":"))
    idx.astype("<u4").tofile(f"{OUT_DIR}/{kind}_idx.bin")
    np.round(np.clip(sim, 0, 1) * 65535).astype("<u2").tofile(f"{OUT_DIR}/{kind}_sc.bin")
    print(f"{kind}: {idx.shape[0]} items, {idx.shape[1]} neighbors each")


def main():
    df = pd.read_csv(SOURCE)
    for col in ("track_id", "track_name", "artists"):
        if col not in df.columns:
            sys.exit(f"Missing column: {col}")
    for col in ("album_name", "track_genre"):
        if col not in df.columns:
            df[col] = ""
    if "popularity" not in df.columns:
        df["popularity"] = 0

    df = df.dropna(subset=["track_id", "track_name", "artists"])
    key = df["track_name"].str.lower().str.strip() + "|" + df["artists"].str.lower().str.strip()
    df = df.assign(_key=key).sort_values("popularity", ascending=False, kind="stable")
    df = df.drop_duplicates(subset="_key").reset_index(drop=True)
    if len(df) < 3:
        sys.exit("Not enough tracks")

    base, genre, genre_names = feature_blocks(df)
    track_matrix = np.hstack([base, GENRE_WEIGHT * genre])
    idx, sim = top_k(track_matrix, K)
    write_kind("tracks", {
        "id": df["track_id"].tolist(),
        "name": df["track_name"].tolist(),
        "artist": df["artists"].str.replace(";", ", ").tolist(),
        "album": df["album_name"].fillna("").tolist(),
        "genre": df["track_genre"].fillna("").tolist(),
        "pop": df["popularity"].astype(int).tolist(),
    }, idx, sim)

    lists = df["artists"].str.split(";")
    rows = np.repeat(np.arange(len(df)), lists.str.len().to_numpy())
    names = pd.Series([a.strip() for group in lists for a in group])
    keep = (names != "").to_numpy()
    rows, names = rows[keep], names[keep].reset_index(drop=True)
    codes, uniques = pd.factorize(names)
    count = np.bincount(codes).astype(np.float32)
    base_sum = np.zeros((len(uniques), base.shape[1]), dtype=np.float32)
    genre_sum = np.zeros((len(uniques), genre.shape[1]), dtype=np.float32)
    np.add.at(base_sum, codes, base[rows])
    np.add.at(genre_sum, codes, genre[rows])
    pop_sum = np.bincount(codes, weights=df["popularity"].to_numpy()[rows])

    selected = count >= MIN_ARTIST_TRACKS
    if selected.sum() < 3:
        sys.exit("Not enough artists")
    count_s = count[selected][:, None]
    artist_base = base_sum[selected] / count_s
    artist_genre = genre_sum[selected] / count_s
    artist_matrix = np.hstack([artist_base, GENRE_WEIGHT * artist_genre])
    idx, sim = top_k(artist_matrix, K)
    top_genre = np.array(genre_names)[genre_sum[selected].argmax(axis=1)]
    write_kind("artists", {
        "name": [u for u, s in zip(uniques, selected) if s],
        "genre": top_genre.tolist(),
        "tracks": count[selected].astype(int).tolist(),
        "pop": np.round(pop_sum[selected] / count[selected]).astype(int).tolist(),
    }, idx, sim)


if __name__ == "__main__":
    main()
