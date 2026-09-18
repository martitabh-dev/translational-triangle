"""
metrics.py

Computation of translational metrics based on the citation network:

- Translational Distance (TD): the minimum number of citation
  "generations" needed to reach a Human article.
- Translational Years (TY): difference in years between the Human
  article reached (via the shortest path) and the source article.
- Translational Closeness (TC): the inverse of TD (0 if there is no
  path).
- Translational Fraction (TF): the fraction of a category's articles
  that eventually reach a Human article.

Algorithmic decision: TD is computed with BFS (breadth-first search)
rather than Dijkstra or some other weighted shortest-path algorithm,
because every citation edge has an implicit weight of 1 (a citation is
a citation, there are no weights to compare). BFS finds the shortest
path in number of edges in O(V+E) time, which is optimal for this case
and avoids the unnecessary cost of maintaining a priority queue.

This module does NOT run the BFS itself (that happens in
forward_citations.py, article by article): here the already-computed
TD/TY/TC/x/y results from papers_full.csv are simply AGGREGATED, to
produce a per-researcher (AuthorID) summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def compute_metrics_by_author(
    df: pd.DataFrame,
    authorid_col: str = "AuthorID",
    status_col: str = "SearchStatus",
    td_col: str = "TD",
    ty_col: str = "TY",
    tc_col: str = "TC",
    ti_col: str = "TI",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Computes the Translational Fraction (TF) and several averages per
    RESEARCHER: for each AuthorID, what proportion of their articles
    eventually reaches a Human article (traversing their citation
    network up to the maximum number of generations the BFS in
    forward_citations.py explores, defined there by the
    _MAX_GENERATIONS constant), and the mean value of several metrics
    over that researcher's set of articles.

    An article "reaches H" if its TD is a finite value.
    forward_citations.py stores TD as float("inf") (not as None/NaN)
    when the BFS exhausts the maximum number of generations without
    finding any H article in the citation tree — that is why np.isfinite
    is used here specifically (which excludes both inf and nan) to
    decide which articles count as "reached", instead of a nullity check
    such as pd.notna.

    Rows whose status_col equals "SKIPPED_INVALID_ROW" (articles
    forward_citations.py could not process because they lacked a valid
    ScopusID and/or Year, see row_is_invalid in that module) are
    excluded from the calculation before grouping, because they have
    neither TD nor the rest of the columns populated and would distort
    the averages.

    Args:
        df: papers_full.csv DataFrame, with TD/TY/TC/x/y already
            computed by compute_forward_citations and
            compute_translational_index.
        authorid_col: name of the column identifying the researcher
            each article belongs to (defaults to "AuthorID").
        status_col: name of the forward-citations search status column;
            used only to exclude rows marked "SKIPPED_INVALID_ROW". If
            the column does not exist in `df`, no filtering is applied
            for this reason.
        td_col: name of the Translational Distance column.
        ty_col: name of the Translational Years column.
        tc_col: name of the Translational Closeness column.
        ti_col: name of the Translational Index column (the geometric
            projection computed by geometry.compute_translational_index).
        x_col: name of the column with each article's x coordinate in
            the triangle (from geometry.compute_positions).
        y_col: name of the column with each article's y coordinate in
            the triangle.

    Returns:
        DataFrame with one row per researcher (sorted by AuthorID), with
        the columns:
          - AuthorNumber: a sequential number assigned to the researcher
            based on the order in which they first appear in `df` (this
            is NOT a stable identifier across different runs if the
            order of the input rows changes; it only serves as a
            readable sequential index within this same results table).
          - AuthorID: the researcher's real identifier (value of
            authorid_col).
          - NArticles: total number of articles for that researcher
            (after excluding SKIPPED_INVALID_ROW).
          - NReachedH: how many of those articles reached H (finite TD).
          - TF: NReachedH / NArticles (Translational Fraction).
          - TDMean: mean of TD, computed ONLY over the articles that
            reached H (if none of them reached it, it is reported as inf
            instead of NaN, to keep the same "no path = inf" convention
            forward_citations.py uses).
          - TYMean: mean of TY, with the same restriction and the same
            inf convention as TDMean.
          - TCMean: mean of TC over ALL of the researcher's articles
            (unlike TDMean/TYMean, this one is not filtered down to
            those that reached H, because TC already equals 0.0 by
            construction for the ones that did not reach it, so
            including them in the mean is correct without needing to
            exclude them separately). If the tc_col column does not
            exist in `df`, 0.0 is reported.
          - TIMean: mean of the Translational Index over all of the
            researcher's articles. If ti_col does not exist in `df`,
            NaN is reported.
          - xMean, yMean: mean of the researcher's x/y coordinates in
            the triangle (the "center of mass" of their articles). If
            the columns do not exist in `df`, they are reported as NaN.
    """

    if status_col in df.columns:
        df = df[df[status_col] != "SKIPPED_INVALID_ROW"]

    result_columns = [
        "AuthorNumber", "AuthorID", "NArticles", "NReachedH", "TF",
        "TDMean", "TYMean", "TCMean", "TIMean", "xMean", "yMean",
    ]
    if df.empty:
        # No articles in this subset (e.g. a year range with no
        # publications): return an empty DataFrame but with the columns
        # already defined, so the rest of the pipeline (to_csv_clean,
        # plot_authors_triangle, _write_summary_report) can keep using
        # it without checking beforehand whether it is empty.
        return pd.DataFrame(columns=result_columns)

    rows = []
    author_id_counter = 0
    for author_id, sub_df in df.groupby(authorid_col):
        author_id_counter += 1
        n_articles = len(sub_df)
        # np.isfinite excludes both inf (BFS exhausted without finding
        # H) and NaN (in case any value is missing); only a TD with a
        # genuine finite number counts as "reached H".
        reached_h_mask = np.isfinite(sub_df[td_col])
        n_reached_h = int(reached_h_mask.sum())
        tf = n_reached_h / n_articles if n_articles > 0 else float("nan")
        reached_h_df = sub_df[reached_h_mask]
        # If none of the researcher's articles reached H, reached_h_df
        # ends up empty, and averaging TD/TY over an empty set makes no
        # sense (.mean() of an empty Series gives NaN); inf is reported
        # instead, consistent with the "no path to H = inf" convention
        # forward_citations.py uses for a single article's TD/TY.
        td_mean = reached_h_df[td_col].mean() if not reached_h_df.empty else float("inf")
        tyMean = reached_h_df[ty_col].mean() if not reached_h_df.empty else float("inf")
        # Unlike TD/TY, TC IS averaged over ALL of the researcher's
        # articles (whether they reached H or not): an article with no
        # path to H already has TC = 0.0 by construction, so including
        # it in the mean is correct and there is no need to filter it
        # out separately.
        tc_mean = sub_df[tc_col].mean() if tc_col in sub_df.columns else 0.0
        ti_mean = sub_df[ti_col].mean() if ti_col in sub_df.columns else float("nan")
        xMean = sub_df[x_col].mean() if x_col in sub_df.columns else float("nan")
        yMean = sub_df[y_col].mean() if y_col in sub_df.columns else float("nan")
        rows.append({
            "AuthorNumber": author_id_counter,
            "AuthorID": author_id,
            "NArticles": n_articles,
            "NReachedH": n_reached_h,
            "TF": tf,
            "TDMean": td_mean,
            "TYMean": tyMean,
            "TCMean": tc_mean,
            "TIMean": ti_mean,
            "xMean": xMean,
            "yMean": yMean,
        })

    return pd.DataFrame(rows).sort_values("AuthorID").reset_index(drop=True)
