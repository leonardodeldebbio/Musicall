(function (root) {
  function normalize(text) {
    return String(text).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
  }

  async function getJson(url, fetchFn) {
    const response = await fetchFn(url);
    if (!response.ok) throw new Error("Failed to load " + url);
    return response.json();
  }

  async function getBuffer(url, fetchFn) {
    const response = await fetchFn(url);
    if (!response.ok) throw new Error("Failed to load " + url);
    return response.arrayBuffer();
  }

  class Index {
    constructor(kind, catalog, idx, sc) {
      this.kind = kind;
      this.catalog = catalog;
      this.idx = idx;
      this.sc = sc;
      this.k = catalog.k;
      this.columns = Object.keys(catalog).filter((key) => Array.isArray(catalog[key]));
      this.names = catalog.name.map(normalize);
      this.artists = catalog.artist ? catalog.artist.map(normalize) : null;
    }

    item(i) {
      const out = { index: i };
      for (const key of this.columns) out[key] = this.catalog[key][i];
      return out;
    }

    search(query, limit = 10) {
      const tokens = normalize(query).split(/\s+/).filter(Boolean);
      if (!tokens.length) return [];
      const joined = tokens.join(" ");
      const found = [];
      for (let i = 0; i < this.names.length; i++) {
        const name = this.names[i];
        const text = this.artists ? name + " " + this.artists[i] : name;
        if (!tokens.every((t) => text.includes(t))) continue;
        let rank = 3;
        if (name === joined) rank = 0;
        else if (name.startsWith(joined)) rank = 1;
        else if (tokens.every((t) => name.includes(t))) rank = 2;
        found.push([rank, i]);
      }
      const pop = this.catalog.pop;
      found.sort((a, b) => a[0] - b[0] || pop[b[1]] - pop[a[1]]);
      return found.slice(0, limit).map((entry) => this.item(entry[1]));
    }

    similar(i, count = 10) {
      const n = Math.max(0, Math.min(count, this.k));
      const out = [];
      for (let j = 0; j < n; j++) {
        const offset = i * this.k + j;
        const entry = this.item(this.idx[offset]);
        entry.score = Math.round((this.sc[offset] / 65535) * 10000) / 10000;
        out.push(entry);
      }
      return out;
    }

    recommend(query, count = 10) {
      const match = this.search(query, 1)[0];
      if (!match) return null;
      return { query: query, match: match, similar: this.similar(match.index, count) };
    }
  }

  async function load(kind, base, fetchFn) {
    const f = fetchFn || root.fetch.bind(root);
    const [catalog, idx, sc] = await Promise.all([
      getJson(base + "/" + kind + ".json", f),
      getBuffer(base + "/" + kind + "_idx.bin", f),
      getBuffer(base + "/" + kind + "_sc.bin", f),
    ]);
    return new Index(kind, catalog, new Uint32Array(idx), new Uint16Array(sc));
  }

  const api = { load: load, normalize: normalize };
  root.Recommender = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
