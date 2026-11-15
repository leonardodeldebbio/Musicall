# Musicall

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge" alt="Development status">
  <img src="https://img.shields.io/badge/License-CC0%201.0-blue?style=for-the-badge" alt="CC0 License">
  <img src="https://img.shields.io/badge/API-Vercel-black?style=for-the-badge&logo=vercel" alt="Vercel">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

<p align="center">
  <b>An open-source music recommendation engine powered by Content-Based Filtering.</b>
</p>

<p align="center">
  Discover similar songs and artists through a simple, reusable API.
</p>

<p align="center">
  <a href="https://leonardodeldebbio.github.io/Musicall/"><b>Live Demo</b></a> ·
  <a href="https://github.com/leonardodeldebbio/Musicall/wiki"><b>Documentation</b></a> ·
  <a href="https://musicall.vercel.app/api/recommend?name=Blinding%20Lights"><b>Try the API</b></a>
</p>

---

## Overview

**Musicall** is an open-source project that provides music and artist recommendations based on their musical characteristics.

The project implements a **Content-Based Filtering** recommendation system that analyses audio features, genres and other musical information to identify similar tracks and artists.

The recommendation engine is built with Python, NumPy and Pandas, while its results are exposed through a REST API hosted on Vercel.

Musicall is designed to be flexible and reusable: its API can be integrated into virtually any application, from small personal projects to complete music discovery and streaming platforms.

## Features

- **Song recommendations:** find tracks similar to a given song.
- **Artist recommendations:** discover artists with similar musical characteristics.
- **Content-Based Filtering:** recommendations based on musical features rather than user listening histories.
- **REST API:** access recommendations through simple HTTP requests.
- **Multiple output formats:** retrieve results in JSON or CSV.
- **Precomputed recommendations:** similarity calculations are performed during dataset generation, reducing API processing requirements.
- **Automated database updates:** GitHub Actions workflows maintain and update the recommendation datasets.
- **Open source:** freely reusable under the CC0 1.0 Universal license.

## How it works

Musicall processes a music dataset originally sourced from Kaggle and transforms its information into numerical representations.

The recommendation algorithm uses several characteristics, including:

- Danceability and energy.
- Acousticness and instrumentalness.
- Tempo and duration.
- Loudness and valence.
- Musical key.
- Music genres.

These features are normalized and combined into numerical vectors. The system then uses **Cosine Similarity** to identify the nearest neighbors for each track and artist.

Recommendations are calculated in advance and stored in optimized JSON and binary files. The API retrieves these precomputed results whenever a request is received.

For a more detailed explanation of the algorithm and data processing pipeline, see the [Recommendation System documentation](https://github.com/leonardodeldebbio/Musicall/wiki/Recommendation-System).

## API

Musicall exposes its recommendation engine through a publicly accessible REST API.

**Base URL:**

```text
https://musicall.vercel.app/api/recommend
```

No API key is currently required.

### Example requests

Find songs similar to a given track:

```http
GET https://musicall.vercel.app/api/recommend?name=Blinding%20Lights
```

Find artists similar to a given artist:

```http
GET https://musicall.vercel.app/api/recommend?name=The%20Weeknd&type=artist
```

Customize the number of recommendations and response format:

```http
GET https://musicall.vercel.app/api/recommend?name=Blinding%20Lights&type=song&limit=5&format=json
```

### Available parameters

| Parameter | Description | Default |
|---|---|---|
| `name` | Song or artist name | Required |
| `type` | `song`, `track` or `artist` | `song` |
| `limit` | Maximum number of recommendations | `10` |
| `format` | `json` or `csv` | `json` |

For complete endpoint documentation, response examples and integration instructions, visit the [Quickstart](https://github.com/leonardodeldebbio/Musicall/wiki/Quickstart).

## Live Demo

You can explore Musicall and experiment with its recommendation capabilities through the live demonstration:

<p align="center">
  <a href="https://leonardodeldebbio.github.io/Musicall/">
    <img src="https://img.shields.io/badge/Explore%20Musicall-Live%20Demo-black?style=for-the-badge" alt="Explore Musicall">
  </a>
</p>

The demonstration is hosted through GitHub Pages and provides an interactive way to discover what the recommendation engine can do.

## Documentation

The project Wiki contains technical documentation covering the API, recommendation algorithm, database and development process.

| Page | Description |
|---|---|
| [Introduction](https://github.com/leonardodeldebbio/Musicall/wiki) | Project overview and objectives |
| [Quickstart](https://github.com/leonardodeldebbio/Musicall/wiki/Quickstart) | API usage and first requests |
| [Recommendation System](https://github.com/leonardodeldebbio/Musicall/wiki/Recommendation-System) | Algorithm and similarity calculations |
| [API Reference](https://github.com/leonardodeldebbio/Musicall/wiki/API-Reference) | API endpoints and parameters |
| [Database & Data Pipeline](https://github.com/leonardodeldebbio/Musicall/wiki/Database-and-Data-Pipeline) | Dataset processing and automated updates |
| [Integration Examples](https://github.com/leonardodeldebbio/Musicall/wiki/Integration-Examples) | Examples for external applications |

*Some documentation sections may still be under development.*

## Technology Stack

Musicall combines several technologies to handle data processing, recommendation generation and API delivery.

| Technology | Purpose |
|---|---|
| Python | Recommendation generation |
| NumPy | Numerical operations and similarity calculations |
| Pandas | Dataset processing and normalization |
| JavaScript | API and recommendation data retrieval |
| Vercel | API hosting and deployment |
| GitHub Actions | Automated workflows and database updates |
| GitHub Pages | Dataset hosting and live demonstration |
| Kaggle | Original music dataset source |

## Project Status

Musicall is currently under active development.

The core recommendation engine and API are available, but additional improvements, optimizations and features are planned.

The project is intended to remain flexible and accessible to developers who want to experiment with music recommendation systems or integrate recommendation capabilities into their own applications.

## Contributing

Contributions, suggestions and feedback are welcome.

If you want to improve Musicall, you can:

- Report bugs or unexpected API behaviour.
- Suggest new features and improvements.
- Improve the recommendation algorithm.
- Contribute to documentation.
- Experiment with alternative datasets and recommendation approaches.

Feel free to explore the source code and open an issue or pull request on GitHub.

## License

Musicall is released under the **CC0 1.0 Universal** public domain dedication.

You are free to copy, modify, distribute and use the project, including for commercial purposes, without asking for permission or providing attribution.

See the [CC0 1.0 Universal license](https://creativecommons.org/publicdomain/zero/1.0/) for further information.
