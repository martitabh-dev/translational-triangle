"""
main.py

This is the project's entry point: the script run from the command line
to run all (or part) of the "translational triangle" pipeline. It
coordinates, in the right order, all the other modules in the package
(scopus_to_pmid.py, mesh_classification.py, geometry.py,
forward_citations.py, metrics.py, visualization.py), but does not itself
implement any heavy business logic: its job is to chain those modules
together and manage a SINGLE results file that grows column by column as
the pipeline progresses.

--------------------------------------------------------------------
THE CENTRAL FILE: papers_full.csv
--------------------------------------------------------------------
The whole pipeline revolves around a single CSV, papers_full.csv, which
is read and re-written IN PLACE at each stage (no new file is generated
per stage). Each stage adds its own columns without touching the ones
already written by earlier stages:

  1. Scopus -> PMID resolution (scopus_to_pmid.py): starting from an
     input CSV with the user's Scopus IDs, resolves each one to its
     PubMed PMID (if it exists), and adds DOI, Title, Status, MeSHList
     and MeSHTerms. This stage CREATES papers_full.csv.

  2. MeSH classification + geometry (mesh_classification.py +
     geometry.py, coordinated here in _classify_papers_full): starting
     from the already-resolved MeSHList/MeSHTerms, adds Category (A/C/H/
     combinations), MeSHTreeNumbers, and each article's geometric x/y/TI
     position within the triangle. It is written AGAIN over
     papers_full.csv, with the previous columns left intact.

  3. Forward citations (forward_citations.py): for each article,
     explores its citation network (who cites it, and in turn who cites
     those citing articles) looking for whether it ever reaches a Human
     article. Adds TD, TY, TC and SearchStatus to papers_full.csv, and
     also generates a separate file, citations_details.csv, with the
     row-by-row detail of every citation explored.

  4. Per-author summary (_run_author_summary, further down in this same
     file): once TD/TY/TC are already computed, groups the articles by
     AuthorID (the researcher identifier) and produces an aggregated
     results CSV (results_by_author.csv), an interactive per-author
     triangle chart (authors_triangle.html) and a text report with cases
     worth reviewing (summary_report.txt).

Because all four stages read and write that same file, the process can
be stopped at any point (for example if an API quota runs out) and
resumed later from where it left off, without losing the work already
done: each stage is designed to detect which columns already exist and
skip the work already performed (see the "resume" logic of each module).

--------------------------------------------------------------------
INPUT CSV FORMAT (the one the user provides)
--------------------------------------------------------------------
The CSV passed with --scopus-input or --resume-scopus must have, at a
minimum, three columns: one with the ScopusID of each article, another
with the identifier of the researcher it belongs to (several articles
can share the same researcher), and a last one with the article's
publication year. Their names in the CSV can be anything; they are
indicated with --scopus-id-column, --author-id-column and
--year-column (default "ScopusID", "AuthorID" and "Year"), and
run_from_scopus_resolution renames them right away to
"ScopusID"/"AuthorID"/"Year", which are the names the rest of the
pipeline uses. Any other column present in the input CSV is dropped in
that same step, before the first call to Scopus: it is not needed for
anything else, so it does not interfere with the rest of the process.
It may be separated by ";" or by "," (detected automatically, see
data_io.read_input_csv): from the point papers_full.csv is generated
onward, the rest of the pipeline is internally unified to ";".

--------------------------------------------------------------------
THE FOUR EXECUTION MODES (mutually exclusive)
--------------------------------------------------------------------
The mode is selected with exactly one of these four flags:

  1. --scopus-input <csv>: full run from scratch. <csv> is the original
     Scopus export (format described above). Runs the 4 stages in
     order, creating papers_full.csv in --output-dir.

  2. --resume-scopus <csv>: same kind of input file as mode 1, but
     meant for when a partial papers_full.csv already exists in
     --output-dir (for example because a previous run stopped halfway
     through the Scopus->PMID resolution when an API key's quota ran
     out). In practice, this mode calls exactly the same function as
     mode 1: convert_scopus_to_pmid's own resume logic already detects
     which ScopusIDs are already resolved in the existing
     papers_full.csv and does not query them again. The distinction
     between --scopus-input and --resume-scopus exists only so it is
     clear from the command line what is being done; the code that runs
     is identical.

  3. --resume-forward <papers_full.csv>: for when the Scopus->PMID
     resolution and the MeSH/geometry classification are ALREADY done
     (papers_full.csv already has PMID, Category, x, y, TI), but forward
     citations have not been computed yet (or were cut off halfway).
     Runs only stage 3 (forward citations) and stage 4 (per-author
     summary).

  4. --authors-summary <papers_full.csv>: for when the ENTIRE pipeline
     has already been computed (including TD/TY/TC) and only the
     per-author summary and its charts need to be regenerated — for
     example after changing something in metrics.py's aggregation
     logic, without needing to repeat any calls to Scopus/PubMed. It is
     the only mode that makes no network requests at all.

Modes 1, 2 and 3 additionally need --scopus-api-key and
--mesh-descriptors, because they involve calls to Scopus/PubMed and/or
MeSH classification. --pubmed-api-key is optional in those same modes
(see the comment next to its argparse.add_argument, further below, for
why). Mode 4 needs none of the three, precisely because it does not
touch the network or reclassify anything.

All three of modes 1, 2 and 3 also run stage 3 (forward citations), so
all three end up with a citations_details.csv, but only mode 3 requires
--citations-details on the command line: in modes 1/2 papers_full.csv
itself always lives inside --output-dir, so citations_details.csv
defaults to that same folder when --citations-details is not given (see
run_from_scopus_resolution); in mode 3, papers_full.csv can be anywhere
(it is a standalone path, not necessarily under --output-dir), so there
is no such folder to default to and it must be given explicitly, the
same way papers_full.csv already is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .data_io import InputError, read_csv_clean, to_csv_clean
from .geometry import compute_positions, compute_translational_index
from .mesh_classification import load_mesh_indices, classify_mesh_codes, get_mesh_tree_numbers
from .metrics import compute_metrics_by_author
from .scopus_to_pmid import convert_scopus_to_pmid
from .visualization import plot_authors_triangle
from .forward_citations import compute_forward_citations

# Special Category labels, distinct from the actual A/C/H combinations
# (e.g. "A", "AH", "ACH"...). They are used both in papers_full.csv and
# to EXCLUDE these rows from the per-category/per-author averages, since
# they do not represent a real translational category:
#   - UNCATEGORIZED: the article WAS successfully resolved (it has a
#     PMID and associated MeSH), but none of its MeSH terms falls under
#     the tree branches that define A, C or H.
#   - NOT_FOUND: the article could never be resolved in Scopus/PubMed at
#     all (Status == "NOT_FOUND" in scopus_to_pmid.py), so there is
#     neither a PMID nor MeSH to try to classify it with.
_UNCATEGORIZED_LABEL = "UNCATEGORIZED"
_NOT_FOUND_LABEL = "NOT_FOUND"

# Single filename used across every stage of the pipeline: the same path
# is read and re-written at each stage (resolution, classification,
# forward citations), inside the output folder given with --output-dir.
PAPERS_FULL_FILENAME = "papers_full.csv"


def _category_to_str(categories: frozenset) -> str:
    """Converts a frozenset of category letters (e.g.
    frozenset({'A', 'H'})) into its text representation, always ordered
    as A-C-H (e.g. "AH", never "HA"), so that the same combination of
    categories always produces the same text regardless of the order in
    which they were detected. An empty frozenset (an article with no
    real category) returns "" — that specific case is handled by
    _label_category, not by this function."""
    order = {"A": 0, "C": 1, "H": 2}
    if not categories:
        return ""
    return "".join(sorted(categories, key=lambda c: order.get(c, 99)))


def _label_category(row: pd.Series) -> str:
    """
    Decides the FINAL text label that goes into the Category column of a
    papers_full.csv row, distinguishing three possible situations:

    - The article was never resolved in Scopus/PubMed at all (that row's
      Status column equals "NOT_FOUND"): the label is directly
      "NOT_FOUND", without even looking at the Category column (which in
      that case will be an empty frozenset, because there was never any
      MeSH to classify).
    - The article WAS resolved, but its Category column (still in
      frozenset form at this point) came out empty because none of its
      MeSH terms fit into the A/C/H branches: the label is
      "UNCATEGORIZED".
    - In any other case: the actual combination of categories found, as
      text (e.g. "A", "AH", "ACH"), using _category_to_str.
    """
    if row["Status"] == _NOT_FOUND_LABEL:
        return _NOT_FOUND_LABEL
    category_str = _category_to_str(row["Category"])
    return category_str if category_str else _UNCATEGORIZED_LABEL


def _write_summary_report(
    df: pd.DataFrame,
    output_dir_path: Path,
    authors_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generates a plain-text report (summary_report.txt) meant to let the
    user quickly review the "odd" or incomplete cases from a pipeline
    run, without having to dig through the full CSV. The report has four
    blocks, in this order:

    1. Researchers with TCMean exactly == 0: none of their articles ever
       reached a Human article by any citation path. It is compared with
       "== 0" and not with "is 0 or empty" on purpose, because TCMean
       only equals exactly 0.0 when it was genuinely computed and came
       out that way (see metrics.py); if the column were missing or had
       NaN it would not match this filter.
    2. Researchers with no position in the triangle (both xMean and
       yMean empty): normally because ALL of their articles are
       UNCATEGORIZED or NOT_FOUND, so there is no valid x/y to average.
       This is the same criterion plot_authors_triangle uses to simply
       skip drawing that researcher's point.
    3. UNCATEGORIZED articles (resolved in Scopus/PubMed, but no MeSH
       term fit into A/C/H), listed by ScopusID/PMID/DOI with no
       duplicates (the same ScopusID appears only once even if it has
       several rows, e.g. from citation generations).
    4. NOT_FOUND articles (never resolved in Scopus/PubMed at all), same
       format as the previous block.

    If `authors_df` is not passed (for example because forward citations
    have not been computed yet), blocks 1 and 2 are simply omitted from
    the report.

    Args:
        df: the papers_full.csv on which UNCATEGORIZED/NOT_FOUND
            articles are looked up.
        output_dir_path: folder where the .txt is written.
        authors_df: the output of compute_metrics_by_author.

    Returns:
        The full text of the report (the same text written to the
        file), in case the caller also wants to use it elsewhere (for
        example printing it to the console).
    """
    lines: list[str] = []

    if authors_df is not None:
        def _author_lines(sub_df: pd.DataFrame) -> list[str]:
            """Formats a row of authors_df as a text line
            "field=value | field=value | ...", using "(empty)" for any
            NaN/None value, and completely skipping any column from the
            list that does not exist in sub_df (so it does not break if
            called with a smaller authors_df)."""
            cols = [
                "AuthorID", "NArticles", "NReachedH", "TF",
                "TDMean", "TYMean", "TCMean", "TIMean", "xMean", "yMean",
            ]
            cols = [c for c in cols if c in sub_df.columns]
            out_lines = []
            for _, row in sub_df.iterrows():
                parts = []
                for col in cols:
                    value = row.get(col)
                    value = value if pd.notna(value) else "(empty)"
                    parts.append(f"{col}={value}")
                out_lines.append("  - " + " | ".join(parts))
            return out_lines

        if "TCMean" in authors_df.columns:
            zero_tc_df = authors_df[authors_df["TCMean"] == 0]
            lines += [
                f"Authors with TC=0 (no article reaches H): {len(zero_tc_df)}",
                *_author_lines(zero_tc_df),
                "",
            ]

        if "xMean" in authors_df.columns and "yMean" in authors_df.columns:
            no_position_df = authors_df[
                authors_df["xMean"].isna() & authors_df["yMean"].isna()
            ]
            lines += [
                f"Authors with no triangle position (xMean/yMean empty): {len(no_position_df)}",
                *_author_lines(no_position_df),
                "",
            ]

    # drop_duplicates(subset="ScopusID") prevents the same article from
    # appearing several times in the listing just because it has several
    # rows in df (e.g. because df is actually citations_details.csv,
    # where the same ScopusID can repeat once per citation generation
    # explored).
    uncategorized_df = df[df["Category"] == _UNCATEGORIZED_LABEL].drop_duplicates(subset="ScopusID")
    not_found_df = df[df["Category"] == _NOT_FOUND_LABEL].drop_duplicates(subset="ScopusID")

    def _identifier_lines(sub_df: pd.DataFrame) -> list[str]:
        """Formats each row as "ScopusID=... | PMID=... | DOI=...",
        replacing any empty identifier with an explanatory text instead
        of leaving it blank."""
        out_lines = []
        for _, row in sub_df.iterrows():
            ScopusID = row["ScopusID"] if pd.notna(row.get("ScopusID")) else "(no ScopusID)"
            pmid = row["PMID"] if pd.notna(row.get("PMID")) else "(no PMID)"
            doi = row["DOI"] if pd.notna(row.get("DOI")) else "(no DOI)"
            out_lines.append(f"  - ScopusID={ScopusID} | PMID={pmid} | DOI={doi}")
        return out_lines

    lines += [
        f"UNCATEGORIZED articles: {len(uncategorized_df)}",
        *_identifier_lines(uncategorized_df),
        "",
        f"NOT_FOUND articles (never resolved in Scopus/PubMed): {len(not_found_df)}",
        *_identifier_lines(not_found_df),
    ]

    report_text = "\n".join(lines)
    report_filename = "summary_report.txt"
    with open(output_dir_path / report_filename, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    return report_text


def _classify_papers_full(df: pd.DataFrame, ui_index: dict, name_index: dict) -> pd.DataFrame:
    """
    Stage 2 of the pipeline: starting from a DataFrame that already has
    MeSHList/MeSHTerms resolved (by scopus_to_pmid.py), adds all the
    columns derived from MeSH classification and triangle geometry:
    Category, MeSHTreeNumbers, x, y, TI. Modifies and returns the same
    DataFrame it received (does not create a copy).

    The order of operations within this function matters:

    1. Category is first computed as a frozenset of letters (e.g.
       frozenset({'A', 'H'})), using classify_mesh_codes row by row. It
       is deliberately stored in this "raw" form (not as text yet)
       because the next step, compute_positions, needs to be able to
       look at which letters each category contains in order to compute
       the corresponding barycenter in geometry.py.
    2. With Category still in frozenset form, each article's x/y
       coordinates are computed (compute_positions) and, from them, the
       Translational Index TI (compute_translational_index).
    3. Only AFTER using the frozenset form for the geometry is Category
       converted to its final text form ("A", "AH", "UNCATEGORIZED",
       "NOT_FOUND", ...) with _label_category, which does need to look
       at the Status column too (not just the set of categories) in
       order to distinguish an article with no category from one that
       was never even resolved.
    4. MeSHTreeNumbers, which get_mesh_tree_numbers returns as an
       in-memory Python list, is flattened to "|"-separated text so it
       can be saved in the CSV (a CSV cannot contain a Python list
       directly). MeSHList, on the other hand, is NOT touched at any
       point in this function: it already arrives from scopus_to_pmid.py
       as "|"-separated text (or None if the article has no MeSH), so it
       is already in the right format to be saved as is.
    5. Finally, UNCATEGORIZED/NOT_FOUND articles have their x/y/TI
       forced to empty: even though compute_positions already assigns
       them NaN automatically (for having no A/C/H letter), this final
       step is an extra, explicit guarantee that no article without a
       real category ends up with a geometric position (which would
       have no meaning) due to some side effect.

    Args:
        df: DataFrame with, at least, the MeSHList, MeSHTerms and Status
            columns already populated.
        ui_index: UI -> TreeNumbers index from mesh_classification.py.
        name_index: Name -> TreeNumbers index from mesh_classification.py,
            used as a fallback when a UI code does not appear in
            ui_index.

    Returns:
        The same DataFrame received, with the Category, x, y, TI and
        MeSHTreeNumbers columns already added/updated.
    """
    df["Category"] = df.apply(
        lambda row: classify_mesh_codes(
            row["MeSHList"], ui_index, name_index=name_index, mesh_terms=row.get("MeSHTerms"),
        ),
        axis=1,
    )
    df["MeSHTreeNumbers"] = df.apply(
        lambda row: get_mesh_tree_numbers(
            row["MeSHList"], ui_index, name_index=name_index, mesh_terms=row.get("MeSHTerms"),
        ),
        axis=1,
    )

    positions = compute_positions(df["Category"])
    df["x"] = positions["x"]
    df["y"] = positions["y"]
    df["TI"] = compute_translational_index(df["x"], df["y"])

    df["Category"] = df.apply(_label_category, axis=1)

    df["MeSHTreeNumbers"] = df["MeSHTreeNumbers"].apply(lambda tns: "|".join(tns) if tns else None)

    no_geometry_mask = df["Category"].isin([_UNCATEGORIZED_LABEL, _NOT_FOUND_LABEL])
    df.loc[no_geometry_mask, ["x", "y", "TI"]] = None

    return df


def _ensure_forward_citation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures papers_full.csv always has the same set of forward-citation
    related columns (TD, TY, TC, SearchStatus), regardless of whether
    that stage has already run or not yet. If any of those columns does
    not exist yet in `df`, it is created empty (None, or "NOT_PROCESSED"
    for SearchStatus, which is the value forward_citations.py expects to
    find for an article that has not been explored yet).

    This is called right after stage 2 (classification), BEFORE stage 3
    (forward citations) has even run once, precisely so that the CSV
    saved right after already has its final column shape
    (PAPERS_FULL_COLUMN_ORDER) from the start, instead of changing shape
    depending on which stages have been run.
    """
    for col in ("TD", "TY", "TC"):
        if col not in df.columns:
            df[col] = None
    if "SearchStatus" not in df.columns:
        df["SearchStatus"] = "NOT_PROCESSED"
    return df


# Final, complete column order for papers_full.csv, as it must always be
# saved to disk (regardless of how many stages have already run).
# Meaning of each column:
#   - AuthorID: identifier of the researcher (comes from the user's
#     input CSV).
#   - ScopusID: the article's Scopus identifier (comes from the input
#     CSV).
#   - Year: the article's publication year.
#   - PMID: the article's PubMed identifier, if it was successfully
#     resolved.
#   - DOI: the article's DOI, if known.
#   - Title: the article's title.
#   - Status: how the PMID was resolved (e.g. "SCOPUS_PMID",
#     "DOI_MATCH", "TITLE_MATCH"), or "NOT_FOUND"/"API_ERROR" if it
#     could not be resolved (see scopus_to_pmid.py).
#   - MeSHList: UI codes of the article's MeSH descriptors, joined with
#     "|".
#   - MeSHTerms: human-readable names of those same MeSH descriptors, in
#     the same order as MeSHList (position by position).
#   - MeSHTreeNumbers: the MeSH tree TreeNumbers found for those
#     descriptors, joined with "|" (see mesh_classification.py).
#   - Category: the article's final translational category ("A", "AH",
#     "ACH", "UNCATEGORIZED" or "NOT_FOUND").
#   - x, y: the article's position within the translational triangle.
#   - TI: the article's Translational Index (see geometry.py).
#   - TD: Translational Distance — number of citation generations to
#     reach a Human article (infinite if it is never reached).
#   - TY: Translational Years — difference in years between this
#     article and the Human article it reached with the shortest path.
#   - TC: Translational Closeness — 1/TD (0 if H is never reached).
#   - SearchStatus: status of the forward-citation search for this
#     article (e.g. "PROCESSED_REACHED_H", "PROCESSED_NOT_REACHED_H",
#     "PROCESSED_NO_CITATIONS", or "NOT_PROCESSED" if that stage has not
#     run yet).
PAPERS_FULL_COLUMN_ORDER = [
    "AuthorID", "ScopusID", "Year", "PMID", "DOI", "Title", "Status",
    "MeSHList", "MeSHTerms", "MeSHTreeNumbers",
    "Category", "x", "y", "TI", "TD", "TY", "TC", "SearchStatus",
]


def _run_author_summary(
    df: pd.DataFrame,
    output_dir_path: Path,
) -> None:
    """
    Stage 4 of the pipeline: generates the per-author summary (CSV +
    interactive chart + text report).

    Requires that TD/TY/TC are already computed (i.e. that stage 3,
    forward citations, has already run); if no article yet has a TD
    value, it is assumed that stage has not run yet and the function
    simply warns on the console and generates nothing, instead of
    failing.

    The result is saved in a subfolder of `output_dir_path`.

    Args:
        df: the complete papers_full.csv (or already filtered
            externally if desired), with TD/TY/TC already computed.
        output_dir_path: folder where the results will be stored.
    """
    print("Starting the results summary, grouped by author")
    if not df["TD"].notna().any():
        print("Warning: TD is not populated (forward citations have not run); "
              "skipping the per-author summary.")
        return

    # Working on a copy so as not to alter the caller's DataFrame, and
    # forcing TD/TY/Year to numeric in case they arrived as text from
    # the CSV (e.g. "inf" as a string instead of float("inf")):
    # pd.to_numeric with errors="coerce" turns any value it cannot
    # interpret into NaN instead of raising an error.
    df = df.copy()
    df["TD"] = pd.to_numeric(df["TD"], errors="coerce")
    df["TY"] = pd.to_numeric(df["TY"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    results_by_author = compute_metrics_by_author(df)
    to_csv_clean(results_by_author, output_dir_path / f"results_by_author.csv")
    plot_authors_triangle(
        results_by_author,
        save_path=str(output_dir_path / f"authors_triangle.html"),
    )
    _write_summary_report(df, output_dir_path, authors_df=results_by_author)
    print(f"Researchers summarized: {len(results_by_author)}")
    print(f"Results saved to: {output_dir_path.resolve()}")


# =====================================
# MODE 1 and 2: full run / resume from the Scopus -> PMID resolution
# =====================================
def run_from_scopus_resolution(
    scopus_input_or_existing: Union[str, Path],
    output_dir: Union[str, Path],
    scopus_api_keys: Union[str, list, tuple],
    mesh_descriptors_xml: Union[str, Path],
    citations_details_csv: Optional[Union[str, Path]] = None,
    pubmed_api_key: Optional[str] = None,
    sleep_time: float = 0.2,
    author_id_column: str = "AuthorID",
    scopus_id_column: str = "ScopusID",
    year_column: str = "Year",
) -> None:
    """
    Implementation shared by modes 1 (--scopus-input) and 2
    (--resume-scopus): both flags end up calling this exact same
    function, with exactly the same arguments.

    - In mode 1, `scopus_input_or_existing` is an original Scopus export
      (with ScopusID/AuthorID/Year columns) and no papers_full.csv
      exists yet in `output_dir`: it is generated from scratch.
    - In mode 2, `scopus_input_or_existing` is that SAME kind of file
      (not a half-finished papers_full.csv), but it is assumed a
      partial papers_full.csv from a previous, interrupted run ALREADY
      exists in `output_dir`. The actual resuming happens inside
      convert_scopus_to_pmid, whose resume=True parameter makes it skip
      any ScopusID that already appears resolved in that existing
      papers_full.csv, and only query Scopus/PubMed for the missing
      ones. Because of this, in practice, running mode 1 on a folder
      that already has previous results HAS the same effect as mode 2:
      the distinction between the two flags exists only to make it
      clearer from the command line what is being done.

    Runs, in order, the four stages described in this file's header,
    always writing to the same papers_full.csv:
      1. ScopusID -> PMID/DOI/Title/Status/MeSH resolution (resume-aware).
      2. MeSH classification + geometry, added to the SAME file.
      3. Forward citations (also generates a separate citations_details.csv).
      4. Final summary per author and year range.

    Args:
        scopus_input_or_existing: path to the Scopus input CSV (same
            format in both modes 1 and 2).
        output_dir: folder where papers_full.csv is created (or already
            exists) along with the rest of the results.
        citations_details_csv: path where citations_details.csv is
            (re)written by stage 3. OPTIONAL here (unlike in mode 3):
            since papers_full.csv itself always lives inside output_dir
            in modes 1/2, if this is not given it defaults to
            output_dir/citations_details.csv, the same folder.
        scopus_api_keys: one or more Elsevier/Scopus API keys (see
            ApiKeyPool in scopus_to_pmid.py for rotation across several).
        pubmed_api_key: NCBI/PubMed E-utilities API key. OPTIONAL:
            without it, PubMed is still queried normally, just limited
            to 3 requests/second (instead of 10) by NCBI.
        mesh_descriptors_xml: path to the official desc*.xml MeSH
            descriptor file, used for A/C/H classification.
        sleep_time: pause in seconds between requests to Scopus/PubMed.
        author_id_column: name, in `scopus_input_or_existing`, of the
            column with the researcher's identifier. Renamed to
            "AuthorID" right away.
        scopus_id_column: name, in `scopus_input_or_existing`, of the
            column with each article's ScopusID. Renamed to "ScopusID"
            right away.
        year_column: name, in `scopus_input_or_existing`, of the column
            with the article's publication year. Renamed to "Year"
            right away. Any other column in the input CSV that is
            neither this one nor author_id_column nor scopus_id_column
            is dropped.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    papers_full_path = output_dir_path / PAPERS_FULL_FILENAME
    if citations_details_csv is None:
        citations_details_csv = output_dir_path / "citations_details.csv"

    # --- Stage 1: ScopusID -> PMID/DOI/Title/MeSH (resume-aware) ---
    # Note: only `sleep_time` is passed here (the pause between requests
    # to Scopus). pubmed_sleep_time is deliberately NOT passed:
    # convert_scopus_to_pmid already computes that value itself,
    # depending on whether pubmed_api_key is provided (0.3 s, ~10 req/s)
    # or not (0.5 s, with margin under NCBI's real 3 req/s limit without
    # a key) — no need to duplicate that logic here.
    convert_scopus_to_pmid(
        input_csv=scopus_input_or_existing,
        output_csv=papers_full_path,
        api_keys=scopus_api_keys,
        pubmed_api_key=pubmed_api_key,
        sleep_time=sleep_time,
        author_id_column=author_id_column,
        scopus_id_column=scopus_id_column,
        year_column=year_column,
    )

    # --- Stage 2: MeSH classification + geometry, written back to the
    # same file ---
    df = read_csv_clean(papers_full_path)
    df = _classify_papers_full(df, *load_mesh_indices(mesh_descriptors_xml))
    df = _ensure_forward_citation_columns(df)
    df = df[PAPERS_FULL_COLUMN_ORDER]
    to_csv_clean(df, papers_full_path)

    # --- Stage 3: forward citations (BFS) ---
    # (same pubmed_sleep_time self-adjustment as in convert_scopus_to_pmid,
    # explained above: no need to pass it here either)
    compute_forward_citations(
        papers_full_csv=papers_full_path,
        mesh_descriptors_xml=mesh_descriptors_xml,
        citations_details_csv=citations_details_csv,
        api_keys=scopus_api_keys,
        pubmed_api_key=pubmed_api_key,
        sleep_time=sleep_time,
    )

    # --- Stage 4: per-author summary ---
    df = read_csv_clean(papers_full_path)
    _run_author_summary(df, output_dir_path)


# =====================================
# MODE 3: resume from forward citations
# =====================================
def run_from_forward_citations(
    papers_full_csv: Union[str, Path],
    output_dir: Union[str, Path],
    citations_details_csv: Union[str, Path],
    scopus_api_keys: Union[str, list, tuple],
    mesh_descriptors_xml: Union[str, Path],
    pubmed_api_key: Optional[str] = None,
    sleep_time: float = 0.2,
) -> None:
    """
    Implementation of mode 3 (--resume-forward): it is assumed the
    received `papers_full_csv` already has PMID/Category/x/y/TI computed
    (i.e. stages 1 and 2 already happened at some earlier point), so
    only stage 3 (forward citations) and stage 4 (per-author summary)
    are run here. The Scopus->PMID resolution and the MeSH
    classification are not touched again.

    Args:
        papers_full_csv: path to an already-classified papers_full.csv
            (with PMID/Category/x/y/TI present).
        output_dir: folder where the rest of this run's results
            (results_by_author.csv, authors_triangle.html,
            summary_report.txt) are saved.
        citations_details_csv: path where citations_details.csv is
            (re)written by stage 3. Not necessarily inside output_dir:
            it is given explicitly, the same way papers_full_csv is.
        scopus_api_keys, pubmed_api_key, mesh_descriptors_xml,
            sleep_time: same parameters as in run_from_scopus_resolution,
            needed because forward citations also query Scopus/PubMed
            and classify MeSH for the new citing articles it discovers.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    papers_full_path = Path(papers_full_csv)

    # (same pubmed_sleep_time self-adjustment as in convert_scopus_to_pmid,
    # see the comment in run_from_scopus_resolution: no need to pass it
    # here either)
    compute_forward_citations(
        papers_full_csv=papers_full_path,
        mesh_descriptors_xml=mesh_descriptors_xml,
        citations_details_csv=citations_details_csv,
        api_keys=scopus_api_keys,
        pubmed_api_key=pubmed_api_key,
        sleep_time=sleep_time,
    )

    df = read_csv_clean(papers_full_path)
    _run_author_summary(df, output_dir_path)


# =====================================
# MODE 4: per-author summary only (no network calls at all)
# =====================================
def summarize_authors(papers_full_csv: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Implementation of mode 4 (--authors-summary): it is assumed the
    received `papers_full_csv` already has the WHOLE pipeline computed,
    including TD/TY/TC (i.e. stages 1, 2 and 3 already completed at some
    earlier point). Here only the per-author summary, its charts and the
    text report are (re)generated — no call to Scopus or PubMed is made
    in this mode. It is the mode meant for when only the
    aggregation/visualization part needs to be regenerated (e.g. after a
    change in metrics.py or visualization.py) without needing to repeat
    any network work already done.

    Args:
        papers_full_csv: path to a papers_full.csv with TD/TY/TC already
            computed.
        output_dir: folder where the aggregated results are saved.

    Raises:
        InputError: if the TD column does not exist or is entirely
            empty, a sign that forward citations were never run on that
            file.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    df = read_csv_clean(papers_full_csv)
    if "TD" not in df.columns or df["TD"].isna().all():
        raise InputError(
            "The given papers_full.csv has no TD computed. Run forward citations first "
            "(--resume-forward or the full pipeline)."
        )
    _run_author_summary(df, output_dir_path)


def _parse_args() -> argparse.Namespace:
    """
    Defines and validates the script's command-line arguments.

    Validation rules applied, in addition to the ones argparse already
    handles on its own (types, default values):

    - EXACTLY one of the four mode flags (--scopus-input,
      --resume-scopus, --resume-forward, --authors-summary) must be
      given: neither zero nor more than one. If this is not satisfied,
      parser.error() itself prints the error message and terminates the
      program (argparse's standard behavior).
    - Modes 1, 2 and 3 need the network (they call Scopus and/or PubMed
      and classify MeSH), so for them --scopus-api-key,
      --pubmed-api-key and --mesh-descriptors are required. Mode 4 does
      not need them at all, precisely because it makes no network call
      and reclassifies nothing.

    Returns:
        The already-validated argparse Namespace, ready to be used
        directly in the `if __name__ == "__main__":` block further
        below.
    """
    parser = argparse.ArgumentParser(
        description="Compute Weber's translational indicators from PubMed/Scopus articles."
    )
    parser.add_argument("--scopus-input", help="MODE 1: raw Scopus export CSV with a Scopus ID column and an author/respondent ID column (names set via --scopus-id-column/--author-id-column/--year-column). Full run.")
    parser.add_argument("--resume-scopus", help="MODE 2: same kind of input as --scopus-input; resumes an existing papers_full.csv in --output-dir.")
    parser.add_argument("--resume-forward", help="MODE 3: path to a papers_full.csv with PMID/Category/x/y/TI already computed.")
    parser.add_argument("--authors-summary", help="MODE 4: path to a papers_full.csv with TD/TY/TC already computed. No network calls.")

    # The CSV passed to --scopus-input/--resume-scopus can name its
    # ScopusID, researcher and year columns however it likes: they are
    # indicated here, and run_from_scopus_resolution renames them to
    # "ScopusID"/"AuthorID"/"Year" (the names the rest of the pipeline
    # uses), dropping any other column the CSV brings. They are only
    # used in modes 1 and 2: in modes 3 and 4 a papers_full.csv is
    # already the starting point, and it already has those columns
    # under their final names.
    parser.add_argument("--author-id-column", default="AuthorID", help="Name of the author/respondent ID column in the raw --scopus-input/--resume-scopus CSV. Renamed internally to 'AuthorID'. Default: AuthorID.")
    parser.add_argument("--scopus-id-column", default="ScopusID", help="Name of the Scopus ID column in the raw --scopus-input/--resume-scopus CSV. Renamed internally to 'ScopusID'. Default: ScopusID.")
    parser.add_argument("--year-column", default="Year", help="Name of the year of publication column in the raw --scopus-input/--resume-scopus CSV. Renamed internally to 'Year'. Default: Year.")

    # --scopus-api-key is REQUIRED: the Scopus API requires a key on
    # EVERY request (there is no endpoint that works without one), and
    # on top of that each key has its own weekly quota that runs out
    # (resetting every 7 days). That is why more than one key,
    # comma-separated, is accepted here: ApiKeyPool (in
    # scopus_to_pmid.py and forward_citations.py) automatically rotates
    # to the next one as soon as one runs out of quota (HTTP 429).
    parser.add_argument("--scopus-api-key", help="One or more Elsevier/Scopus API keys, comma-separated. Required for modes 1-3.")
    # --pubmed-api-key, on the other hand, is OPTIONAL: PubMed (NCBI
    # E-utilities) works the same without it, with no endpoint blocked,
    # just with the request limit dropping from 10/s to 3/s. Unlike
    # Scopus, there is no weekly quota to run out of, so a single key
    # already gives the maximum speed indefinitely: there is no need to
    # rotate between several as with Scopus. The "10/s with a key, 3/s
    # without one" is automatically translated by
    # convert_scopus_to_pmid/compute_forward_citations into their pause
    # between requests (0.3 s with a key, 0.5 s without it) — see the
    # comment next to its call, further below.
    parser.add_argument("--pubmed-api-key", help="NCBI/PubMed E-utilities API key. Optional: without it, requests are still "
                              "made, just capped at 3/s instead of 10/s (pubmed_sleep_time adjusts itself "
                              "accordingly, see run_from_scopus_resolution).")
    parser.add_argument("--mesh-descriptors", help="Path to the NLM desc*.xml MeSH descriptor file. Required for modes 1-3.")

    parser.add_argument("--sleep-time", type=float, default=0.2, help="Pause (s) between Scopus/PubMed requests.")
    parser.add_argument("--output-dir", default="results", help="Output folder for all results.")
    # Explicit path to citations_details.csv, the same way papers_full.csv
    # is given explicitly in mode 3 (--resume-forward). REQUIRED only for
    # mode 3: in modes 1/2 papers_full.csv itself always lives inside
    # --output-dir, so citations_details.csv defaults to that same
    # folder (see run_from_scopus_resolution) when this is not given.
    # In mode 3, papers_full.csv can be anywhere, so there is no such
    # folder to default to and it must be given explicitly.
    parser.add_argument("--citations-details", help="Path to citations_details.csv. Required for mode 3 (--resume-forward). "
                              "Optional for modes 1/2 (--scopus-input/--resume-scopus): defaults to "
                              "<output-dir>/citations_details.csv. Either way, it is read, if it already "
                              "exists, to resume, and (re)written with the updated detail rows.")

    args = parser.parse_args()

    modes_given = [args.scopus_input, args.resume_scopus, args.resume_forward, args.authors_summary]
    if sum(m is not None for m in modes_given) != 1:
        parser.error("Specify exactly one of --scopus-input, --resume-scopus, --resume-forward, --authors-summary.")

    needs_network = args.scopus_input or args.resume_scopus or args.resume_forward
    if needs_network:
        if not args.scopus_api_key:
            parser.error("--scopus-api-key is required for this mode.")
        # --pubmed-api-key is OPTIONAL: without it,
        # run_from_scopus_resolution / run_from_forward_citations still
        # call PubMed normally, just limited to 3 req/s (instead of 10
        # req/s) by NCBI.
        if not args.mesh_descriptors:
            parser.error("--mesh-descriptors is required for this mode.")
        # --citations-details is required only for mode 3
        # (--resume-forward): in modes 1/2 it defaults to
        # <output-dir>/citations_details.csv when not given (see
        # run_from_scopus_resolution), since papers_full.csv itself
        # always lives inside --output-dir in those two modes.
        if args.resume_forward and not args.citations_details:
            parser.error("--citations-details is required for mode 3 (--resume-forward).")
    return args


if __name__ == "__main__":
    # Actual entry point when running "python -m scripts.main ...": the
    # arguments are parsed and dispatched to the corresponding mode. The
    # four modes are mutually exclusive (already enforced by
    # _parse_args), so at most one of these branches runs.
    args = _parse_args()

    try:
        if args.authors_summary:
            summarize_authors(papers_full_csv=args.authors_summary, output_dir=args.output_dir)

        elif args.resume_forward:
            run_from_forward_citations(
                papers_full_csv=args.resume_forward,
                output_dir=args.output_dir,
                citations_details_csv=args.citations_details,
                scopus_api_keys=args.scopus_api_key,
                pubmed_api_key=args.pubmed_api_key,
                mesh_descriptors_xml=args.mesh_descriptors,
                sleep_time=args.sleep_time,
            )

        else:
            # Modes 1 (--scopus-input) and 2 (--resume-scopus) share
            # exactly the same implementation: convert_scopus_to_pmid's
            # resume logic (resume=True) already takes care of skipping
            # any ScopusID a previous run already left resolved in
            # papers_full.csv, so no extra code is needed here to tell
            # "from scratch" apart from "resuming".
            source = args.scopus_input or args.resume_scopus
            run_from_scopus_resolution(
                scopus_input_or_existing=source,
                output_dir=args.output_dir,
                citations_details_csv=args.citations_details,
                author_id_column=args.author_id_column,
                scopus_id_column=args.scopus_id_column,
                year_column=args.year_column,
                scopus_api_keys=args.scopus_api_key,
                pubmed_api_key=args.pubmed_api_key,
                mesh_descriptors_xml=args.mesh_descriptors,
                sleep_time=args.sleep_time,
            )
    except InputError as exc:
        # User-side configuration/format errors (columns that do not
        # exist in the input CSV, a papers_full.csv with no TD computed,
        # etc.): reported with a clear message and the program exits
        # with code 1, WITHOUT showing the Python traceback (which adds
        # nothing here and would only confuse, giving the impression
        # there is a bug in the script itself).
        print(f"\nError: {exc}\n", file=sys.stderr)
        sys.exit(1)