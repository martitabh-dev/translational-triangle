# Translational Triangle

A Python pipeline that measures how *translational* a body of biomedical research is, following the "translational triangle" framework proposed by Weber (2013). Starting from a researcher's list of publications (identified by Scopus ID), the pipeline resolves each article in PubMed, classifies it along the Animal / Cell / Human axis based on its MeSH descriptors, places it geometrically inside a triangle to compute a single translational score, and then follows its citation network forward in time to see whether — and how quickly — that research eventually gets cited by human-subject work.

> This repository includes only the pipeline code and a small worked example (`dataset/test.csv`, ~100 rows, with its corresponding output in `results/`), used to demonstrate and validate that the pipeline runs correctly end to end. It does not include the full input dataset or the full results of any particular research run.

## Contents

- [Background](#background)
- [How the pipeline works](#how-the-pipeline-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Input data](#input-data)
- [Usage](#usage)
- [Output files](#output-files)
- [Metrics glossary](#metrics-glossary)
- [Project structure](#project-structure)
- [Resuming interrupted runs](#resuming-interrupted-runs)
- [API quota handling](#api-quota-handling)
- [Reference](#reference)

## Background

Translational research moves along a continuum: from purely preclinical work (animal models, cell and molecular biology) toward research that directly involves human subjects. Weber's translational triangle framework represents this continuum geometrically, as an equilateral triangle with three vertices:

- **A (Animal)** — research on non-human organisms.
- **C (Cell)** — cellular, molecular and microbiological research.
- **H (Human)** — research directly involving human subjects.

Every article is classified against this framework using its MeSH descriptors, then placed at a position inside the triangle that reflects the mixture of A/C/H content it contains. That position collapses to a single number, the **Translational Index (TI)**, which measures how far an article sits along the "preclinical → human" axis.

Beyond an article's own content, the pipeline also tracks translational **impact**: it follows the citation network forward from each article to see whether it is ever cited — directly or through a chain of citations — by an article in the Human category, and how long that takes. This is what the **TD / TY / TC / TF** metrics capture (see the [glossary](#metrics-glossary) below).

## How the pipeline works

The entire pipeline revolves around a single CSV file, `papers_full.csv`, which is read and rewritten *in place* at each stage — no new file is generated per stage, and each stage only adds its own columns without touching what earlier stages already wrote. This is what makes the pipeline resumable: it can be stopped at any point (for example, if an API quota runs out) and picked up again later without losing work already done.

The pipeline has four stages:

1. **Scopus → PMID resolution** (`scopus_to_pmid.py`). Starting from a CSV of Scopus IDs, resolves each article to its PubMed identifier (PMID) through a three-step cascade (direct PMID from Scopus, then DOI match, then title match in PubMed), and downloads its DOI, title and MeSH descriptors. This stage *creates* `papers_full.csv`.
2. **MeSH classification + geometry** (`mesh_classification.py` + `geometry.py`). Classifies each article into one or more of the A/C/H categories based on where its MeSH descriptors sit in the official MeSH tree, then computes its (x, y) position inside the triangle and its Translational Index (TI).
3. **Forward citations** (`forward_citations.py`). For each article, explores its citation network generation by generation (breadth-first search) to determine whether it is ever cited by an H-category article, producing TD, TY, TC, and a detailed `citations_details.csv` log of every citation explored.
4. **Per-researcher summary** (coordinated from `main.py`, using `metrics.py` and `visualization.py`). Aggregates the per-article metrics by researcher (`AuthorID`), producing a summary CSV, an interactive HTML chart, and a text report flagging cases worth reviewing.

Because every stage reads and writes the same `papers_full.csv`, `main.py` exposes four mutually exclusive run modes so the pipeline can be entered at whichever stage is needed — see [Usage](#usage).

## Requirements

- Python 3.9 or newer.
- The packages listed in `requirements.txt`:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `requests`
- Internet access to the Scopus (Elsevier) and PubMed (NCBI E-utilities) APIs, for the modes that query them.
- An Elsevier/Scopus API key (required for stages 1–3).
- An NCBI/PubMed E-utilities API key (optional, but strongly recommended — see [API quota handling](#api-quota-handling)).
- The official NLM MeSH descriptor file (`desc*.xml`), downloadable from the [NLM MeSH XML site](https://www.nlm.nih.gov/mesh/xmlmesh.html). A copy (`desc2026.xml`) is expected under `dataset/`.

## Setup

```bash
pip install -r requirements.txt
```

The scripts are organized as a package (`scripts/`) and are meant to be run as a module from the project root, e.g. `python -m scripts.main ...` (see [Usage](#usage)).

## Input data

The CSV passed to `--scopus-input` / `--resume-scopus` must contain, at minimum, three columns:

- A **Scopus ID** for each article.
- An **author/researcher identifier** — several articles can share the same researcher.
- The article's **publication year**.

The column names can be anything; they are mapped with `--scopus-id-column`, `--author-id-column` and `--year-column` (defaulting to `ScopusID`, `AuthorID` and `Year`). Any other column present in the input file is dropped before processing starts. The file may be separated by `;` or by `,` — the separator is detected automatically.

`dataset/test.csv` is the small (~100-row) example input included in this repository, used throughout this README to illustrate the format and the commands below:

```
AuthorID,ScopusID,Year
R_scmttKhnvJjJcnT,84859741988,2011
R_wRipgHE7S3haP97,80053124273,2011
R_3HztxEZK5AOhfD3,80055024880,2011
```

## Usage

All commands are run as a module from the project root. Exactly one of the four mode flags must be given.

### Mode 1 — full run from scratch

```bash
python -m scripts.main --scopus-input dataset/test.csv \
    --scopus-api-key SCOPUS_API_KEY \
    --pubmed-api-key PUBMED_API_KEY \
    --mesh-descriptors dataset/desc2026.xml \
    --output-dir results/
```

Runs all four stages in order and creates `papers_full.csv` in `--output-dir`. Running this exact command against the example `dataset/test.csv` is what produced the `results/` folder included in this repository.

### Mode 2 — resume the Scopus → PMID resolution

```bash
python -m scripts.main --resume-scopus dataset/test.csv \
    --scopus-api-key SCOPUS_API_KEY \
    --pubmed-api-key PUBMED_API_KEY \
    --mesh-descriptors dataset/desc2026.xml \
    --output-dir results/
```

Same kind of input file as mode 1, used when a partial `papers_full.csv` already exists in `--output-dir` (for example, because a previous run stopped when an API key's quota ran out). In practice this calls the exact same code as mode 1 — the resolution logic already detects which Scopus IDs are resolved and skips them; the separate flag exists only to make the intent explicit from the command line.

### Mode 3 — resume from forward citations

```bash
python -m scripts.main --resume-forward results/papers_full.csv \
    --scopus-api-key SCOPUS_API_KEY \
    --pubmed-api-key PUBMED_API_KEY \
    --mesh-descriptors dataset/desc2026.xml \
    --output-dir results/
```

For when the Scopus → PMID resolution and MeSH/geometry classification are already done (`papers_full.csv` already has `PMID`, `Category`, `x`, `y`, `TI`), but forward citations have not been computed yet, or were cut off partway through. Runs only stage 3 and stage 4.

### Mode 4 — regenerate the per-researcher summary only

```bash
python -m scripts.main --authors-summary results/papers_full.csv \
    --output-dir results/
```

For when the entire pipeline (including `TD`/`TY`/`TC`) is already computed and only the aggregation and charts need to be regenerated — for example after a change to the aggregation logic. This is the only mode that makes **no network requests at all**.

### Common options

| Flag | Description | Default |
|---|---|---|
| `--scopus-api-key` | One or more Elsevier/Scopus API keys, comma-separated. Required for modes 1–3. | — |
| `--pubmed-api-key` | NCBI/PubMed E-utilities API key. Optional — without it, requests are still made, just capped at 3/s instead of 10/s. | — |
| `--mesh-descriptors` | Path to the NLM `desc*.xml` MeSH descriptor file. Required for modes 1–3. | — |
| `--author-id-column` | Name of the researcher-ID column in the raw input CSV. | `AuthorID` |
| `--scopus-id-column` | Name of the Scopus-ID column in the raw input CSV. | `ScopusID` |
| `--year-column` | Name of the publication-year column in the raw input CSV. | `Year` |
| `--sleep-time` | Pause, in seconds, between Scopus/PubMed requests. | `0.2` |
| `--output-dir` | Output folder for all results. | `results` |

## Output files

All output is written to `--output-dir`. The `results/` folder included in this repository contains exactly these files, produced by running the pipeline against the small example dataset described in [Input data](#input-data):

Two files below (`papers_full.csv` and `citations_details.csv`) carry a `Category`/`CitingCategory` column that, besides a real A/C/H combination, can also hold one of two special labels for articles that don't get a translational classification at all:

- **`NOT_FOUND`** — the article could not be resolved to a PubMed record by any of the three methods the pipeline tries (a PMID already reported by Scopus, a DOI match, or a title match in PubMed — see stage 1 in [How the pipeline works](#how-the-pipeline-works)). With no PMID, there is no MeSH data to classify it with, so it never gets an A/C/H category.
- **`UNCATEGORIZED`** — the opposite situation: the article *was* resolved to a PubMed record (it has a PMID), but still ends up with no A/C/H category, for one of two reasons: either it has MeSH descriptors but none of them falls under the Animal, Cell or Human branches of the MeSH tree, so it doesn't fit anywhere in the triangle; or it has no MeSH descriptors at all in PubMed, even though the record itself exists (this can happen with records PubMed hasn't indexed with MeSH yet).

Neither case gets a triangle position (`x`/`y`/`TI` are left empty) and neither counts toward the per-researcher category/position averages, but neither is discarded either: the article still keeps its row in `papers_full.csv`, and — when it shows up as a citation rather than as an original article — it still keeps its row in `citations_details.csv`, since even an unclassifiable citation can lead to an H-category article further down its own citation chain. `summary_report.txt` lists every `NOT_FOUND`/`UNCATEGORIZED` article found in a run, precisely so they're easy to review (see below).

- **`papers_full.csv`** — the central results file, one row per article, with the following columns:

  | Column | Meaning |
  |---|---|
  | `AuthorID` | Researcher identifier, from the input CSV. |
  | `ScopusID` | The article's Scopus identifier. |
  | `Year` | The article's publication year. |
  | `PMID` | The article's PubMed identifier, if resolved. |
  | `DOI` | The article's DOI, if known. |
  | `Title` | The article's title. |
  | `Status` | How the PMID was resolved (`SCOPUS_PMID`, `DOI_MATCH`, `TITLE_MATCH`), or `NOT_FOUND`/`API_ERROR`. |
  | `MeSHList` | UI codes of the article's MeSH descriptors, joined with `\|`. |
  | `MeSHTerms` | Human-readable names of those same descriptors, in the same order as `MeSHList`. |
  | `MeSHTreeNumbers` | MeSH tree numbers found for those descriptors, joined with `\|`. |
  | `Category` | Final translational category: an A/C/H combination (e.g. `A`, `AH`, `ACH`), or `UNCATEGORIZED`/`NOT_FOUND` (see explanation above). |
  | `x`, `y` | The article's position inside the translational triangle. |
  | `TI` | Translational Index. |
  | `TD` | Translational Distance. |
  | `TY` | Translational Years. |
  | `TC` | Translational Closeness. |
  | `SearchStatus` | Status of the forward-citation search for this article (`NOT_PROCESSED`, `SKIPPED_INVALID_ROW`, `PROCESSED_REACHED_H`, `PROCESSED_NOT_REACHED_H`, or `PROCESSED_NO_CITATIONS`). |

- **`citations_details.csv`** — the detailed, row-by-row log behind every TD/TY/TC value in `papers_full.csv`: one row per *(original article) → (citing article found while exploring its forward-citation network)* relationship, generated by stage 3 (`forward_citations.py`). An original article can have anywhere from zero to dozens of rows here, one per citation actually explored across up to 4 generations. Columns:

  | Column | Meaning |
  |---|---|
  | `ScopusID` | Scopus ID of the **original** article this citation search started from (matches a row in `papers_full.csv`). |
  | `PMID` | PubMed ID of that same original article. |
  | `Generation` | Which citation "hop" this row was found at: `1` means it directly cites the original article, `2` means it cites something that cites the original, and so on, up to the pipeline's maximum of 4 generations. |
  | `ParentNodeID` | Identifier of the node whose own citations were being explored when this row turned up — its PMID, or `SCOPUS:<ScopusID>` if it has none. For a `Generation` `1` row, this is the original article itself. |
  | `ParentPMID` | PMID of that parent node, if it has one. |
  | `ParentScopusID` | Scopus ID of that parent node. |
  | `CitingNodeID` | Identifier of the citing article that was found — its PMID, or `SCOPUS:<ScopusID>` if it could not be resolved to a PMID. |
  | `CitingScopusID` | Scopus ID of the citing article. |
  | `CitingDOI` | DOI of the citing article, if known. |
  | `CitingPMID` | PubMed ID of the citing article, if it was resolved (reusing the same PMID-resolution cascade as `scopus_to_pmid.py`). |
  | `CitingYear` | Publication year of the citing article. |
  | `CitingMeSHList` | UI codes of the citing article's MeSH descriptors, joined with `\|`. |
  | `CitingMeSHTerms` | Human-readable names of those same descriptors, in the same order as `CitingMeSHList`. |
  | `CitingMeSHTreeNumbers` | MeSH tree numbers of those descriptors, joined with `\|`. |
  | `CitingCategory` | Translational category of the citing article: an A/C/H combination, or `UNCATEGORIZED`/`NOT_FOUND` (see explanation above). |
  | `PathToH` | `True` only for the rows that lie on the actual shortest citation path from the original article to the first (oldest) `H`-category article found — the same path that determines that article's `TD`/`TY` in `papers_full.csv`. `False` for every other citation that was explored but is not part of that path. |

- **`results_by_author.csv`** — the per-researcher aggregation computed by `metrics.py`, one row per `AuthorID`:

  | Column | Meaning |
  |---|---|
  | `AuthorNumber` | A sequential number assigned to each researcher based on the order they first appear in the input. It is only a readable index within this results table — not a stable identifier across different runs — and it is also what gets printed directly on each point's label in `authors_triangle.html`'s manual-selection mode (see below), since the full `AuthorID` would be unreadable at that scale. |
  | `AuthorID` | The researcher's real identifier, from the input CSV. |
  | `NArticles` | Total number of that researcher's articles (after excluding rows skipped for lacking a valid `ScopusID`/`Year`). |
  | `NReachedH` | How many of those articles reached an `H`-category article by some citation path (i.e. have a finite `TD`). |
  | `TF` | Translational Fraction: `NReachedH / NArticles`. |
  | `TDMean` | Mean Translational Distance, computed only over the articles that reached H (reported as `inf`, not `NaN`, if none of them did). |
  | `TYMean` | Mean Translational Years, with the same restriction and the same `inf` convention as `TDMean`. |
  | `TCMean` | Mean Translational Closeness over **all** of the researcher's articles (unlike `TDMean`/`TYMean`, this is not restricted to the ones that reached H, since an article that never reaches H already contributes exactly `0.0` by construction). |
  | `TIMean` | Mean Translational Index over all of the researcher's articles. |
  | `xMean`, `yMean` | Mean x/y position of the researcher's articles inside the triangle — effectively the "center of mass" of their body of work, and what `authors_triangle.html` plots for each researcher. |

- **`authors_triangle.html`** — a self-contained, interactive chart (built with Plotly, loaded from a CDN) plotting each researcher at their `(xMean, yMean)` position inside the triangle.

  Only researchers with a valid triangle position are plotted at all — see the `NOT_FOUND`/`UNCATEGORIZED` explanation above for why a researcher might not have one (it happens when every one of their articles falls into one of those two labels, leaving nothing to average into an `(x, y)` position). The page shows a banner right below its title with the count of researchers left out this way, so their absence isn't mistaken for a bug.

  Among the researchers who *are* plotted:
  - Each point is sized proportionally to the square root of `NArticles`, so it's the point's *area* — not its diameter — that reflects the number of articles.
  - Each point is colored on a purple (low `TCMean`) → green (high `TCMean`) gradient. Researchers whose `TCMean` is `0` (none of their articles ever reached an H-category article) or unavailable are drawn in neutral gray instead, outside the color scale; the legend panel next to the chart shows a running count of how many researchers fall into that gray group.
  - `AuthorNumber` (see the `results_by_author.csv` columns above) is what gets printed directly on a point's label, since the full `AuthorID` would be unreadable at this scale; it's also accepted, alongside `AuthorID`, as a search term.

  It supports two view modes, switchable from the page itself: a "Top N" slider showing the N researchers with the most articles (drawn without on-point labels, to keep the chart readable), and a manual-selection mode to look up and pin specific researchers by `AuthorID` or `AuthorNumber`, each shown labeled by its `AuthorNumber` on the chart.

- **`summary_report.txt`** — a plain-text report meant to let you review the "odd" or incomplete cases from a run without digging through the CSVs by hand, generated by `main.py`. It has up to four blocks, in this order:

  1. Researchers whose `TCMean` is exactly `0` — none of their articles ever reached an H-category article by any citation path.
  2. Researchers with no position in the triangle at all (`xMean` and `yMean` both empty) — normally because every one of their articles ended up `UNCATEGORIZED` or `NOT_FOUND`.
  3. Every `UNCATEGORIZED` article (successfully resolved in PubMed, but none of its MeSH terms falls under the A/C/H branches), listed once each (by `ScopusID`/`PMID`/`DOI`), even if it appears on several rows elsewhere (e.g. as a citation found in more than one generation).
  4. Every `NOT_FOUND` article (never resolved in Scopus/PubMed at all), listed the same way.

  Blocks 1 and 2 are included only when the per-researcher summary has already been computed (i.e. `results_by_author.csv` exists for that run).

## Metrics glossary

- **Category (A / C / H)** — the translational-triangle category (or categories) an article's MeSH descriptors place it in. An article can belong to more than one at once (e.g. `AH` for a study combining an animal model with human patients); there is no priority between categories.
- **Translational Index (TI)** — a single number summarizing an article's own position along the "preclinical → human" axis of the triangle, computed from its (x, y) coordinates.
- **Translational Distance (TD)** — the minimum number of citation "generations" (hops) needed, starting from this article, to reach an article in the Human category. Infinite if that article is never reached.
- **Translational Years (TY)** — the difference in years between this article and the Human-category article it reached via the shortest citation path (the same one that determines its TD).
- **Translational Closeness (TC)** — `1 / TD`, so a shorter citation path to Human research gives a value closer to 1, and an article that never reaches H gets a TC of 0.
- **Translational Fraction (TF)** — at the researcher level, the proportion of a researcher's articles that eventually reach some Human-category article, regardless of how long that citation path is.

## Project structure

```
scripts/
├── __init__.py              Package overview and framework summary.
├── main.py                  Entry point / CLI: coordinates all four stages.
├── data_io.py                CSV reading/writing helpers with consistent dtype handling.
├── scopus_to_pmid.py         Stage 1: Scopus -> PMID/DOI/Title/MeSH resolution.
├── mesh_classification.py    Stage 2 (part 1): MeSH-tree-based A/C/H classification.
├── geometry.py                Stage 2 (part 2): triangle geometry and Translational Index.
├── forward_citations.py      Stage 3: forward-citation BFS (TD/TY/TC).
├── metrics.py                  Stage 4: per-researcher aggregation (TF and means).
└── visualization.py           Stage 4: interactive triangle chart (authors_triangle.html).
```

## Resuming interrupted runs

Because every stage reads and writes the same `papers_full.csv`, and each stage is designed to detect which of its own columns are already populated, the pipeline can be safely stopped and resumed at any point:

- Stage 1 (`scopus_to_pmid.py`) checkpoints its progress every 25 new rows, and its `resume=True` default (used by modes 1 and 2) skips any Scopus ID already resolved in an existing `papers_full.csv`.
- Stage 3 (`forward_citations.py`) likewise checkpoints and reuses previously explored citation data.
- Modes 2, 3 and 4 exist precisely to re-enter the pipeline at the right stage without repeating work — or network calls — already done.

## API quota handling

Both the Scopus and PubMed APIs enforce request limits. As soon as either one responds with HTTP 429 (quota/rate limit exhausted), the pipeline **stops immediately** — it does not retry or wait — after saving everything resolved up to that point to `papers_full.csv` and printing the quota's reset time to the console, if the API reports it. Simply rerunning the same command (with `--resume-scopus` or `--resume-forward`, as appropriate) picks the work back up where it left off.

- **Scopus** requires an API key on every request and each key has its own weekly quota. `--scopus-api-key` accepts several keys, comma-separated: the pipeline automatically rotates to the next one as soon as the current one is exhausted, retrying only once every available key has failed.
- **PubMed** works without an API key, just at a lower rate (3 requests/second instead of 10). A single `--pubmed-api-key` is enough to unlock the higher rate indefinitely — there is no weekly quota to rotate around.

## Reference

Weber, G. M. (2013). *Identifying translational science within the triangle of biomedicine.* AMIA Summits on Translational Science Proceedings. (See `Identifying translational science within the triangle of biomedicine - Weber (2013).pdf` in the project root.)

---

<p align="center">
  <img src="assets/momentum-footer.png" alt="Marta Benítez Hernández — Proyecto MOMENTUM — INGENIO (CSIC–UPV) — mbenher1@upcnet.upv.es — Universitat Politècnica de València, Camino de Vera, s/n, 46022, Valencia — Ministerio de Ciencia, Innovación y Universidades, CSIC, Universitat Politècnica de València, Ingenio (CSIC-UPV), MaX" width="480">
</p>
