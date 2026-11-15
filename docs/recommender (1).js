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
      this.n = Number.isInteger(catalog.n) ? catalog.n : catalog.name.length;
      this.families = Array.isArray(catalog.families) ? catalog.families : null;
      this.columns = Object.keys(catalog).filter(
        (key) => key !== "families" && Array.isArray(catalog[key]) && catalog[key].length === this.n
      );
      this.names = null;
      this.artists = null;
      this.familyKeys = this.families ? this.families.map(normalize) : null;
      this.genreKeys = new Map();
    }

    prepare() {
      if (this.names) return;
      this.names = this.catalog.name.map(normalize);
      this.artists = this.catalog.artist ? this.catalog.artist.map(normalize) : null;
    }

    item(i) {
      const out = { index: i };
      for (const key of this.columns) out[key] = this.catalog[key][i];
      if (this.families && out.family !== undefined) {
        out.family_id = out.family;
        out.family = this.families[out.family_id] || "";
      }
      if (out.audio !== undefined) out.audio = out.audio === 1;
      return out;
    }

    genreKey(i) {
      const raw = this.catalog.genre[i];
      let key = this.genreKeys.get(raw);
      if (key === undefined) {
        key = normalize(raw);
        this.genreKeys.set(raw, key);
      }
      return key;
    }

    matcher(options) {
      const o = options || {};
      const genre = o.genre ? normalize(o.genre) : "";
      const family = o.family ? normalize(o.family) : "";
      const minYear = Number.isFinite(o.minYear) ? o.minYear : null;
      const maxYear = Number.isFinite(o.maxYear) ? o.maxYear : null;
      const minPop = Number.isFinite(o.minPopularity) ? o.minPopularity : null;
      const audioOnly = o.audioOnly === true;
      if (!genre && !family && minYear === null && maxYear === null && minPop === null && !audioOnly) {
        return null;
      }
      const c = this.catalog;
      return (i) => {
        if (genre && (!c.genre || this.genreKey(i) !== genre)) return false;
        if (family) {
          if (!c.family || !this.familyKeys) return false;
          if (this.familyKeys[c.family[i]] !== family) return false;
        }
        if (minYear !== null || maxYear !== null) {
          const year = c.year ? c.year[i] : 0;
          if (!year) return false;
          if (minYear !== null && year < minYear) return false;
          if (maxYear !== null && year > maxYear) return false;
        }
        if (minPop !== null && (!c.pop || c.pop[i] < minPop)) return false;
        if (audioOnly && (!c.audio || c.audio[i] !== 1)) return false;
        return true;
      };
    }

    search(query, limit = 10, options) {
      const tokens = normalize(query).split(/\s+/).filter(Boolean);
      if (!tokens.length) return [];
      this.prepare();
      const match = this.matcher(options);
      const joined = tokens.join(" ");
      const found = [];
      for (let i = 0; i < this.names.length; i++) {
        const name = this.names[i];
        const text = this.artists ? name + " " + this.artists[i] : name;
        if (!tokens.every((t) => text.includes(t))) continue;
        if (match && !match(i)) continue;
        let rank = 3;
        if (name === joined) rank = 0;
        else if (name.startsWith(joined)) rank = 1;
        else if (tokens.every((t) => name.includes(t))) rank = 2;
        found.push([rank, i]);
      }
      const pop = this.catalog.pop || [];
      found.sort((a, b) => a[0] - b[0] || (pop[b[1]] || 0) - (pop[a[1]] || 0));
      return found.slice(0, limit).map((entry) => this.item(entry[1]));
    }

    similar(i, count = 10, options) {
      if (!Number.isInteger(i) || i < 0 || i >= this.n) return [];
      const limit = Math.max(0, count);
      const match = this.matcher(options);
      const out = [];
      for (let j = 0; j < this.k && out.length < limit; j++) {
        const offset = i * this.k + j;
        const target = this.idx[offset];
        if (match && !match(target)) continue;
        const entry = this.item(target);
        entry.score = Math.round((this.sc[offset] / 65535) * 10000) / 10000;
        out.push(entry);
      }
      return out;
    }

    recommend(query, count = 10, options) {
      const match = this.search(query, 1)[0];
      if (!match) return null;
      return { query: query, match: match, similar: this.similar(match.index, count, options) };
    }

    browse(options, limit = 10) {
      const match = this.matcher(options);
      const out = [];
      for (let i = 0; i < this.n && out.length < limit; i++) {
        if (!match || match(i)) out.push(this.item(i));
      }
      return out;
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

  async function loadCategories(base, fetchFn) {
    const f = fetchFn || root.fetch.bind(root);
    return getJson(base + "/categories.json", f);
  }

  const api = { load: load, loadCategories: loadCategories, normalize: normalize };
  root.Recommender = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
