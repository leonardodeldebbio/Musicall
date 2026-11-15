# Musicall

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.1.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-CC0-555555?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Hosted%20on-Vercel-000000?style=flat-square&logo=vercel" alt="Vercel">
</p>

<p align="center">
  <strong>An open-source music recommendation engine.</strong>
  <br>
  Discover similar songs and artists through audio analysis and content-based filtering.
</p>

<p align="center">
  <a href="https://leonardodeldebbio.github.io/Musicall/"><img src="https://img.shields.io/badge/LIVE%20DEMO-181717?style=for-the-badge&logo=github&logoColor=white" alt="Live Demo"></a>
  <a href="https://github.com/leonardodeldebbio/Musicall/wiki"><img src="https://img.shields.io/badge/DOCUMENTATION-0969DA?style=for-the-badge&logo=gitbook&logoColor=white" alt="Documentation"></a>
  <a href="https://musicall.vercel.app/api/recommend?name=Blinding%20Lights"><img src="https://img.shields.io/badge/REST%20API-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="API"></a>
</p>

---

## Overview

**Musicall** is an open-source music recommendation system designed to identify similarities between songs and artists.

It implements a **Content-Based Filtering** approach, analysing musical characteristics, audio features and genre information to generate recommendations without relying on users' listening histories.

The recommendation engine is developed in Python, while its results are exposed through a public REST API.

## Features

- **Song recommendations:** Discover tracks with similar musical characteristics.
- **Artist recommendations:** Identify artists with comparable musical styles.
- **Audio snippet analysis:** Extract additional musical information from short audio excerpts to improve recommendation accuracy.
- **Content-Based Filtering:** Generate recommendations using musical characteristics rather than user preferences or listening histories.
- **REST API:** Access recommendation results through HTTP requests.
- **Expanded dataset:** An evolving collection of tracks and musical information.
- **Automated updates:** GitHub Actions workflows maintain and update the recommendation database.

## Recommendation System

Musicall processes a music dataset and transforms relevant musical information into numerical representations.

The system considers several audio features, including:

- Danceability and energy.
- Acousticness and instrumentalness.
- Tempo and loudness.
- Valence and duration.
- Musical genres and other track characteristics.

These features are processed and compared using **Cosine Similarity**, allowing the system to identify tracks with similar musical profiles.

Audio snippet analysis provides an additional source of information by extracting characteristics directly from short audio excerpts, helping improve the accuracy of recommendations.

Recommendations are precomputed and stored in optimized files, allowing the API to retrieve results efficiently.

For further technical details, see the [Recommendation System documentation](https://github.com/leonardodeldebbio/Musicall/wiki/Recommendation-System).

## REST API

Musicall provides a publicly accessible REST API. No API key is currently required.

**Base URL**

```text
https://musicall.vercel.app/api/recommend
```

**Example requests**

```http
GET /api/recommend?name=Blinding%20Lights
```

```http
GET /api/recommend?name=The%20Weeknd&type=artist
```

**Available parameters**

| Parameter | Description | Default |
|---|---|---|
| `name` | Song or artist name | Required |
| `type` | `song`, `track` or `artist` | `song` |
| `limit` | Maximum number of recommendations | `10` |
| `format` | `json` or `csv` | `json` |

For complete endpoint documentation and integration examples, visit the [API Quickstart](https://github.com/leonardodeldebbio/Musicall/wiki/Quickstart).

## Documentation

The project Wiki contains additional technical information about the system, its architecture and implementation.

- [Introduction](https://github.com/leonardodeldebbio/Musicall/wiki)
- [Quickstart](https://github.com/leonardodeldebbio/Musicall/wiki/Quickstart)
- [Recommendation System](https://github.com/leonardodeldebbio/Musicall/wiki/Recommendation-System)

## Changelog

### v1.1.0 — Recommendation Improvements

- Introduced audio snippet analysis to extract musical characteristics directly from short audio excerpts.
- Integrated snippet-based analysis into the recommendation process to improve prediction accuracy.
- Expanded the original dataset with additional tracks and musical information.

### v1.0.0 — Initial Release

- First public release of Musicall.
- Introduced the Content-Based Filtering recommendation engine.
- Implemented song and artist recommendations.
- Published the REST API and live demonstration.
- Established the original music dataset and automated processing pipeline.

## License

Musicall is released under the **[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)** license.

The project can be freely used, modified, distributed and integrated into other applications, including commercial projects.
