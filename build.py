import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

BUILD_VERSION = 3

SOURCE = os.environ.get("SOURCE_CSV", "data/tracks.csv")
OUT_DIR = os.environ.get("OUT_DIR", "docs/data")
K = int(os.environ.get("NEIGHBORS", "20"))
CANDIDATE_FACTOR = int(os.environ.get("CANDIDATE_FACTOR", "4"))
ARTIST_CAP = int(os.environ.get("ARTIST_CAP", "3"))
GENRE_FAMILIES = int(os.environ.get("GENRE_FAMILIES", "12"))
MIN_ARTIST_TRACKS = int(os.environ.get("MIN_ARTIST_TRACKS", "1"))
MEM_MB = int(os.environ.get("MEM_MB", "1200"))

WEIGHTS = {
    "audio": float(os.environ.get("W_AUDIO", "1.0")),
    "tech": float(os.environ.get("W_TECH", "0.5")),
    "harm": float(os.environ.get("W_HARM", "0.4")),
    "flag": float(os.environ.get("W_FLAG", "0.3")),
    "year": float(os.environ.get("W_YEAR", "0.6")),
    "pop": float(os.environ.get("W_POP", "0.3")),
    "genre": float(os.environ.get("W_GENRE", "0.8")),
    "family": float(os.environ.get("W_FAMILY", "0.5")),
}
GENRE_BONUS = float(os.environ.get("GENRE_BONUS", "0.10"))
ALBUM_BONUS = float(os.environ.get("ALBUM_BONUS", "0.15"))
ARTIST_BONUS = float(os.environ.get("ARTIST_BONUS", "0.10"))

GROUPS = {
    "audio": ["danceability", "energy", "valence", "acousticness",
              "instrumentalness", "liveness", "speechiness"],
    "tech": ["loudness", "tempo", "duration_ms"],
    "harm": ["mode", "time_signature"],
    "flag": ["explicit"],
}
LOG_COLUMNS = {"speechiness", "acousticness", "instrumentalness", "liveness"}

CONFIG = {
    "version": BUILD_VERSION, "k": K, "candidate_factor": CANDIDATE_FACTOR,
    "artist_cap": ARTIST_CAP, "weights": WEIGHTS, "families": GENRE_FAMILIES,
    "genre_bonus": GENRE_BONUS, "album_bonus": ALBUM_BONUS,
    "artist_bonus": ARTIST_BONUS, "min_artist_tracks": MIN_ARTIST_TRACKS,
}
OUTPUT_FILES = [
    "tracks.json", "tracks_idx.bin", "tracks_sc.bin",
    "artists.json", "artists_idx.bin", "artists_sc.bin", "categories.json",
]


def unit(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def top_k(matrix, k, normalize=True):
    x = np.ascontiguousarray(matrix, dtype=np.float32)
    if normalize:
        x = unit(x)
    n = len(x)
    k = min(k, n - 1)
    chunk = int(max(32, min(4096, MEM_MB * 1e6 / (12.0 * n))))
    idx = np.empty((n, k), dtype=np.int32)
    sim = np.empty((n, k), dtype=np.float32)
    rows = np.arange(chunk)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = x[start:end] @ x.T
        block[rows[: end - start], np.arange(start, end)] = -np.inf
        part = np.argpartition(block, n - k, axis=1)[:, n - k:]
        vals = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[start:end] = np.take_along_axis(part, order, axis=1)
        sim[start:end] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def rerank(idx, sim, genre_codes, album_codes, art, step=20000):
    n, c = idx.shape
    top = 1.0 + GENRE_BONUS + ALBUM_BONUS + ARTIST_BONUS
    out_idx = np.empty_like(idx)
    out_sim = np.empty_like(sim)
    for start in range(0, n, step):
        end = min(start + step, n)
        bi = idx[start:end]
        bs = sim[start:end].astype(np.float32)
        rows = np.arange(start, end)[:, None]
        bs += GENRE_BONUS * (genre_codes[bi] == genre_codes[rows])
        own_album = album_codes[rows]
        bs += ALBUM_BONUS * ((album_codes[bi] == own_album) & (own_album >= 0))
        mine = art[start:end][:, None, :, None]
        theirs = art[bi][:, :, None, :]
        shared = ((mine == theirs) & (mine >= 0)).any(axis=(2, 3))
        bs += ARTIST_BONUS * shared
        order = np.argsort(-bs, axis=1, kind="stable")
        out_idx[start:end] = np.take_along_axis(bi, order, axis=1)
        out_sim[start:end] = np.clip(np.take_along_axis(bs, order, axis=1) / top, 0, 1)
    return out_idx, out_sim


def cap_per_artist(idx, sim, artist_codes, k, cap, step=20000):
    n, c = idx.shape
    k = min(k, c)
    if cap <= 0:
        return idx[:, :k].copy(), sim[:, :k].copy()
    out_idx = np.empty((n, k), dtype=idx.dtype)
    out_sim = np.empty((n, k), dtype=sim.dtype)
    position = np.arange(c)[None, :]
    for start in range(0, n, step):
        end = min(start + step, n)
        block_idx = idx[start:end]
        block_sim = sim[start:end]
        codes = artist_codes[block_idx]
        order = np.argsort(codes, axis=1, kind="stable")
        sorted_codes = np.take_along_axis(codes, order, axis=1)
        is_new = np.ones(sorted_codes.shape, dtype=bool)
        is_new[:, 1:] = sorted_codes[:, 1:] != sorted_codes[:, :-1]
        run_start = np.maximum.accumulate(np.where(is_new, position, 0), axis=1)
        rank_sorted = position - run_start
        rank = np.empty_like(rank_sorted)
        np.put_along_axis(rank, order, rank_sorted, axis=1)
        allowed = rank < cap
        first = np.argsort(~allowed, axis=1, kind="stable")[:, :k]
        out_idx[start:end] = np.take_along_axis(block_idx, first, axis=1)
        out_sim[start:end] = np.take_along_axis(block_sim, first, axis=1)
    return out_idx, out_sim


def load_tracks():
    df = pd.read_csv(SOURCE, low_memory=False)
    for col in ("track_id", "track_name", "artists"):
        if col not in df.columns:
            sys.exit(f"Missing column: {col}")
    for col in ("album_name", "track_genre"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "popularity" not in df.columns:
        df["popularity"] = 0
    df = df.dropna(subset=["track_id", "track_name", "artists"])
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    key = (
        df["track_name"].astype(str).str.lower().str.strip()
        + "|"
        + df["artists"].astype(str).str.lower().str.strip()
    )
    df = df.assign(_key=key).sort_values("popularity", ascending=False, kind="stable")
    df = df.drop_duplicates(subset="_key").drop_duplicates(subset="track_id")
    return df.reset_index(drop=True)


def standardize(x):
    x = x.astype("float64")
    x = x.fillna(x.median())
    std = x.std()
    if not np.isfinite(std) or std < 1e-9:
        return None
    return ((x - x.mean()) / std).clip(-4, 4).to_numpy(np.float32)


def feature(df, col):
    if col not in df.columns:
        return None
    if col == "explicit":
        x = df[col].astype(str).str.lower().isin(["true", "1"]).astype("float64")
    else:
        x = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if col in LOG_COLUMNS:
        x = np.log10(x.clip(lower=0) + 0.01)
    if col == "duration_ms":
        x = np.log1p(x.clip(lower=0))
    return standardize(x)


def year_block(df):
    if "year" not in df.columns:
        return np.zeros(len(df), np.float32), np.zeros(len(df), np.float32), False
    y = pd.to_numeric(df["year"], errors="coerce")
    if y.notna().sum() == 0:
        return np.zeros(len(df), np.float32), np.zeros(len(df), np.float32), False
    filled = y.fillna(y.median())
    std = filled.std()
    z = ((filled - filled.mean()) / std).clip(-4, 4) if std > 0 else filled * 0.0
    return z.to_numpy(np.float32), filled.to_numpy(np.float32), True


def build_groups(df, year_z, has_year):
    groups = {}
    for name, columns in GROUPS.items():
        parts = [feature(df, c) for c in columns]
        parts = [p for p in parts if p is not None]
        if name == "harm" and "key" in df.columns:
            key = pd.to_numeric(df["key"], errors="coerce").fillna(0).clip(0, 11).to_numpy(np.float32)
            angle = 2 * np.pi * key / 12
            parts += [0.7 * np.sin(angle), 0.7 * np.cos(angle)]
        if parts:
            groups[name] = np.stack(parts, axis=1).astype(np.float32)
    pop = standardize(df["popularity"])
    if pop is not None:
        groups["pop"] = pop[:, None]
    if has_year:
        groups["year"] = year_z[:, None]
    if "audio" not in groups:
        sys.exit("No usable audio features")
    return groups


def cluster_genres(centroids, counts, target):
    g = len(centroids)
    target = max(1, min(target, g))
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -np.inf)
    active = np.ones(g, dtype=bool)
    sizes = np.ones(g)
    members = [[i] for i in range(g)]
    while active.sum() > target:
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        merged = (sizes[i] * sim[i] + sizes[j] * sim[j]) / (sizes[i] + sizes[j])
        sim[i, :] = merged
        sim[:, i] = merged
        sim[i, i] = -np.inf
        sizes[i] += sizes[j]
        members[i] += members[j]
        active[j] = False
    family_of = np.zeros(g, dtype=np.int32)
    names = []
    for number, i in enumerate(np.flatnonzero(active)):
        group = members[i]
        lead = max(group, key=lambda m: counts[m])
        names.append(lead)
        for m in group:
            family_of[m] = number
    return family_of, names


def classify(df, core):
    genres = df["track_genre"].str.strip().replace("", "unknown")
    codes, names = pd.factorize(genres, sort=True)
    names = [str(x) for x in names]
    g = len(names)
    n = len(df)
    onehot = sparse.csr_matrix((np.ones(n, np.float32), (np.arange(n), codes)), shape=(n, g))
    counts = np.bincount(codes, minlength=g)
    centroids = np.asarray(onehot.T @ core) / counts[:, None]
    centroids = unit(centroids.astype(np.float32))
    family_of, family_lead = cluster_genres(centroids, counts, GENRE_FAMILIES)
    family_names = [names[i] for i in family_lead]
    return codes, names, counts, onehot, centroids, family_of, family_names


def dense_matrix(groups, centroid_block, family_block):
    blocks = [
        WEIGHTS[name] * g / np.sqrt(g.shape[1]) for name, g in groups.items()
    ]
    blocks.append(WEIGHTS["genre"] * centroid_block)
    blocks.append(WEIGHTS["family"] * family_block)
    return unit(np.hstack(blocks).astype(np.float32))


def artist_matrix(df, m=5):
    lists = df["artists"].astype(str).str.split(";")
    mapping = {}
    out = np.full((len(df), m), -1, dtype=np.int64)
    for i, group in enumerate(lists):
        for j, name in enumerate(group[:m]):
            name = name.strip().lower()
            if name:
                out[i, j] = mapping.setdefault(name, len(mapping))
    primary = out[:, 0].copy()
    blank = primary < 0
    primary[blank] = len(mapping) + np.arange(int(blank.sum()))
    return out, primary


def album_codes(df):
    album = df["album_name"].str.lower().str.strip()
    key = album + "|" + df["artists"].astype(str).str.split(";").str[0].str.lower().str.strip()
    codes, _ = pd.factorize(key)
    codes = codes.astype(np.int64)
    codes[(album == "").to_numpy()] = -1
    return codes


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_kind(kind, catalog, idx, sim):
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog["n"] = int(idx.shape[0])
    catalog["k"] = int(idx.shape[1])
    with open(f"{OUT_DIR}/{kind}.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, separators=(",", ":"))
    idx.astype("<u4").tofile(f"{OUT_DIR}/{kind}_idx.bin")
    np.round(np.clip(sim, 0, 1) * 65535).astype("<u2").tofile(f"{OUT_DIR}/{kind}_sc.bin")
    print(f"{kind}: {idx.shape[0]} items, {idx.shape[1]} neighbors each", flush=True)


def report(idx, genre_codes, family_codes, years, pop, artist_codes, album):
    own = np.arange(len(idx))[:, None]
    print(
        "Quality: same genre {:.3f} | same family {:.3f} | mean year gap {:.2f} | "
        "same artist {:.3f} | same album {:.3f} | recommended popularity {:.1f} (catalog {:.1f})".format(
            (genre_codes[idx] == genre_codes[own]).mean(),
            (family_codes[idx] == family_codes[own]).mean(),
            np.abs(years[idx] - years[own]).mean(),
            (artist_codes[idx] == artist_codes[own]).mean(),
            ((album[idx] == album[own]) & (album[own] >= 0)).mean(),
            pop[idx].mean(),
            pop.mean(),
        ),
        flush=True,
    )


def main():
    df = load_tracks()
    n = len(df)
    if n < 3:
        sys.exit("Not enough tracks")

    fingerprint = hashlib.sha256(
        json.dumps({"config": CONFIG, "csv": file_digest(SOURCE)}, sort_keys=True).encode()
    ).hexdigest()
    marker = os.path.join(OUT_DIR, "build.json")
    if os.path.exists(marker) and all(os.path.exists(os.path.join(OUT_DIR, f)) for f in OUTPUT_FILES):
        with open(marker, encoding="utf-8") as f:
            if json.load(f).get("fingerprint") == fingerprint:
                print("Inputs unchanged, recommendation data is up to date", flush=True)
                return

    pop = df["popularity"].to_numpy(np.float32)
    year_z, year_raw, has_year = year_block(df)
    groups = build_groups(df, year_z, has_year)
    core = np.hstack([g for name, g in groups.items() if name != "pop"])

    genre_codes, genre_names, genre_counts, onehot, centroids, family_of, family_names = classify(df, core)
    family_codes = family_of[genre_codes]
    family_block = np.eye(len(family_names), dtype=np.float32)[family_codes]
    dense = dense_matrix(groups, centroids[genre_codes], family_block)

    art, primary = artist_matrix(df)
    album = album_codes(df)

    candidates = min(n - 1, K * CANDIDATE_FACTOR)
    idx, sim = top_k(dense, candidates, normalize=False)
    idx, sim = rerank(idx, sim, genre_codes, album, art)
    idx, sim = cap_per_artist(idx, sim, primary, K, ARTIST_CAP)
    report(idx, genre_codes, family_codes, year_raw, pop, primary, album)

    write_kind("tracks", {
        "id": df["track_id"].tolist(),
        "name": df["track_name"].tolist(),
        "artist": df["artists"].astype(str).str.replace(";", ", ").tolist(),
        "album": df["album_name"].tolist(),
        "genre": df["track_genre"].tolist(),
        "year": year_raw.astype(int).tolist() if has_year else [0] * n,
        "family": family_codes.tolist(),
        "pop": pop.astype(int).tolist(),
        "families": family_names,
    }, idx, sim)
    del idx, sim

    lists = df["artists"].astype(str).str.split(";")
    counts_per_track = lists.str.len().to_numpy()
    rows = np.repeat(np.arange(n), counts_per_track)
    names = np.array([a.strip() for group in lists for a in group], dtype=object)
    keep = names != ""
    rows, names = rows[keep], names[keep]
    codes, uniques = pd.factorize(names)
    a_count = np.bincount(codes)
    weight = pop[rows].astype(np.float64) + 1.0
    membership = sparse.csr_matrix((weight, (codes, rows)), shape=(len(uniques), n))
    wsum = np.asarray(membership.sum(axis=1)).ravel()

    selected = a_count >= MIN_ARTIST_TRACKS
    if selected.sum() < 3:
        sys.exit("Not enough artists")
    profile = np.asarray(membership @ dense) / wsum[:, None]
    genre_mass = np.asarray((membership @ onehot).todense())
    top_genre = genre_mass.argmax(axis=1)
    artist_year = np.asarray(membership @ year_raw) / wsum
    artist_pop = np.bincount(codes, weights=pop[rows]) / a_count

    profile = profile[selected]
    top_genre = top_genre[selected]
    a_idx, a_sim = top_k(profile, K, normalize=True)
    a_pop = artist_pop[selected]
    a_family = family_of[top_genre]
    own = np.arange(len(a_idx))[:, None]
    print(
        "Artist quality: same genre {:.3f} | same family {:.3f}".format(
            (top_genre[a_idx] == top_genre[own]).mean(),
            (a_family[a_idx] == a_family[own]).mean(),
        ),
        flush=True,
    )
    write_kind("artists", {
        "name": [u for u, s in zip(uniques, selected) if s],
        "genre": [genre_names[g] for g in top_genre],
        "family": a_family.tolist(),
        "tracks": a_count[selected].astype(int).tolist(),
        "pop": np.round(a_pop).astype(int).tolist(),
        "year": np.round(artist_year[selected]).astype(int).tolist() if has_year else [0] * len(a_idx),
        "families": family_names,
    }, a_idx, a_sim)

    genre_pop = np.bincount(genre_codes, weights=pop) / genre_counts
    genre_year = np.bincount(genre_codes, weights=year_raw) / genre_counts
    decades = (year_raw // 10 * 10).astype(int)
    eras = {}
    if has_year:
        values, counts = np.unique(decades, return_counts=True)
        eras = {f"{int(v)}s": int(c) for v, c in zip(values, counts)}
    categories = {
        "genres": [
            {
                "name": genre_names[i],
                "family": int(family_of[i]),
                "tracks": int(genre_counts[i]),
                "pop": round(float(genre_pop[i]), 1),
                "year": round(float(genre_year[i]), 1) if has_year else 0,
            }
            for i in range(len(genre_names))
        ],
        "families": [
            {
                "name": family_names[f],
                "genres": [genre_names[i] for i in range(len(genre_names)) if family_of[i] == f],
                "tracks": int(genre_counts[family_of == f].sum()),
            }
            for f in range(len(family_names))
        ],
        "eras": eras,
    }
    with open(os.path.join(OUT_DIR, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, separators=(",", ":"))
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint}, f)
    print("Build complete", flush=True)


if __name__ == "__main__":
    main()
