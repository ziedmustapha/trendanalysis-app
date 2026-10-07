# trendanalysis-app

PubMed trend analysis UI: fetch papers, cluster related work with SPECTER embeddings, then name clusters and extract trends with a **local Ollama model**.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In a second terminal, start Ollama (once per machine boot):

```bash
~/.local/bin/ollama serve
```

Then:

```bash
python main.py
```

Open [http://localhost:5050](http://localhost:5050).

The first **Analyze** click downloads the `allenai/specter` embedding model. The first cluster-name / trends call loads `qwen2.5:3b` into memory and can take a little longer.

## Recommended parameters

Settings that produce well-separated, publication-quality clusters:

| Field | Value | Why |
| --- | --- | --- |
| Query | `food safety AND microbiome` | Spans several distinct subfields, so clusters are genuinely different |
| Max | `150` | The single biggest lever. Low values starve the older-paper pool and clusters become arbitrary |
| Present year | `2024` | Needs enough older papers behind it to compare against |
| Strength of cluster fusion | `0.97` | See below |

SPECTER cluster-to-cluster cosine similarities sit between roughly **0.85 and 0.99**, so the
fusion slider only does useful work above ~0.93 — anything lower merges everything into a single
blob. The slider is therefore restricted to the 0.90–1.00 range. Rough guide:

- `0.95` — few, broad clusters (3–4)
- `0.97` — balanced, usually 5–6 clusters (recommended)
- `0.99` — many small clusters (8–10)

Measured cluster separation (mean inter-centroid distance over mean intra-cluster spread) for a
few queries at `max=150`:

| Query | Best threshold | Clusters | Separation |
| --- | --- | --- | --- |
| `food safety AND (mycotoxin OR blockchain OR allergen)` | 0.95 | 3 | 3.27 |
| `food safety AND microbiome` | 0.97 | 5 | 2.48 |
| `food safety` | 0.95 | 4 | 2.18 |
| `antimicrobial resistance AND food` | 0.98 | 7 | 2.14 |

Raising `Max` from 60 to 150 roughly doubled separation, because each new paper then finds
genuinely similar older papers instead of whatever happened to be returned.

## Example runs

Each of these was run end to end; screenshots of the full UI and of the 3D plot alone are in
[`screenshots/`](screenshots).

| # | Query | Max | Year | Fusion | Clusters | Character | Files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recommended | `food safety AND microbiome` | 150 | 2024 | 0.97 | 5 | Balanced; the default | `zia-*` |
| A | `food safety AND (mycotoxin OR blockchain OR allergen)` | 150 | 2024 | 0.95 | 3 | Cleanest separation of all | `config-a-mycotoxin-allergen-*` |
| B | `food safety` | 200 | 2024 | 0.95 | 5 | Broad query, visibly dispersed | `config-b-broad-foodsafety-*` |
| C | `antimicrobial resistance AND food` | 150 | 2024 | 0.98 | 7 | Fine-grained, still coherent | `config-c-amr-finegrained-*` |
| D | `food safety AND microbiome` | 150 | 2024 | 0.99 | 8 | Over-split (duplicate themes) | `config-d-microbiome-099-*` |
| E | `food safety AND microbiome` | 150 | 2023 | 0.97 | 4 | Same query, earlier year | `config-e-microbiome-2023-*` |

Two things these runs show beyond the numbers. Config A is the best figure because its query
deliberately spans three unrelated subfields, so the clusters are genuinely distinct rather than
slices of one topic. Config B is the cautionary case: a bare `food safety` query produces a
lopsided cloud with one cluster far from the rest, which is why a compound query is worth the
effort.

Config D also shows the cost of pushing fusion to `0.99` — it splits gut-microbiome work into
both "Gut Health and Metabolism Papers" and "Gut Health and Metabolite Regulation Papers", which
are the same theme.

## Known limitations

- `find_closest_papers` uses only the **first 10** new papers, whatever `Max` is set to. The
  figures therefore summarise 10 papers from the present year and their nearest older neighbours,
  not the full result set. Worth stating in a figure caption.
- Clusters holding a single new paper (8 related) often make `qwen2.5:3b` emit placeholder trend
  text with empty bullets. Larger clusters, or a larger model, produce usable trends. This shows
  up in configs B, C and D.
- The separation figures above come from a custom metric (mean inter-centroid distance over mean
  intra-cluster spread), not a standard index like silhouette score. Use them comparatively.

## Local LLM

This machine is a MacBook Air M4 with 16 GB RAM, so the default model is **Qwen 2.5 3B** (`qwen2.5:3b`). Pull it once:

```bash
~/.local/bin/ollama pull qwen2.5:3b
```

Copy `.env.example` to `.env` if you do not already have one. `LLM_PROVIDER=ollama` is the default.

## Optional cloud LLM

A Claude Pro chat plan cannot be used as an API. If you later add an Anthropic Console key, set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic`.

`.env` is gitignored and must never be committed.

## Optional Node frontend proxy

The Flask app already serves the UI. If you prefer the original Express proxy:

```bash
cd frontend
npm install
npm start
```

Then open [http://localhost:3000](http://localhost:3000) with Flask still running on port 5050.
