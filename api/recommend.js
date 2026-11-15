const Recommender = require("../docs/recommender.js");

const DATA_URL = (process.env.DATA_URL || "https://leonardodeldebbio.github.io/Musicall/data").replace(/\/$/, "");
const TTL_MS = 60 * 60 * 1000;
const cache = {};

function getIndex(kind) {
  const hit = cache[kind];
  if (hit && Date.now() - hit.time < TTL_MS) return hit.promise;
  const promise = Recommender.load(kind, DATA_URL, fetch);
  cache[kind] = { time: Date.now(), promise: promise };
  promise.catch(() => {
    if (cache[kind] && cache[kind].promise === promise) delete cache[kind];
  });
  return promise;
}

function first(value) {
  return Array.isArray(value) ? value[0] : value;
}

function shape(kind, item, rank) {
  const out = { rank: rank };
  for (const [key, value] of Object.entries(item)) {
    if (key === "index") continue;
    out[key === "pop" ? "popularity" : key] = value;
  }
  if (kind === "tracks") out.url = "https://open.spotify.com/track/" + item.id;
  return out;
}

function csvCell(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

function toCsv(rows) {
  const columns = [];
  for (const row of rows) for (const key of Object.keys(row)) if (!columns.includes(key)) columns.push(key);
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((c) => csvCell(row[c])).join(","));
  return lines.join("\n") + "\n";
}

function send(res, status, body, type) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Content-Type", type || "application/json; charset=utf-8");
  if (status === 200) res.setHeader("Cache-Control", "public, s-maxage=3600, stale-while-revalidate=600");
  res.status(status).send(typeof body === "string" ? body : JSON.stringify(body, null, 2));
}

module.exports = async function handler(req, res) {
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
    res.status(204).end();
    return;
  }

  const q = req.query || {};
  const typeParam = String(first(q.type) || "song").toLowerCase();
  const name = String(first(q.name) || "").trim();
  const format = String(first(q.format) || "json").toLowerCase();
  const limitParam = parseInt(first(q.limit), 10);

  let kind;
  if (typeParam === "song" || typeParam === "track") kind = "tracks";
  else if (typeParam === "artist") kind = "artists";
  else return send(res, 400, { error: "Invalid type. Use song or artist." });

  if (!name) return send(res, 400, { error: "Missing name parameter." });
  if (format !== "json" && format !== "csv") return send(res, 400, { error: "Invalid format. Use json or csv." });

  let index;
  try {
    index = await getIndex(kind);
  } catch (error) {
    return send(res, 502, { error: "Could not load the data files. Check DATA_URL and that the workflow has run." });
  }

  const limit = Math.min(Math.max(Number.isFinite(limitParam) ? limitParam : 10, 1), index.k);
  const result = index.recommend(name, limit);
  if (!result) return send(res, 404, { error: "No " + (kind === "tracks" ? "song" : "artist") + " found for: " + name });

  const match = shape(kind, result.match, 0);
  const results = result.similar.map((item, i) => shape(kind, item, i + 1));

  if (format === "csv") return send(res, 200, toCsv([match].concat(results)), "text/csv; charset=utf-8");
  send(res, 200, { type: kind === "tracks" ? "song" : "artist", query: name, limit: limit, match: match, results: results });
};
