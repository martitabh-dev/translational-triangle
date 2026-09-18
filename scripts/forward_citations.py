"""
forward_citations.py

Gets the forward citations of each original article via the Scopus
Search API (query=REF(ScopusID)), resolves the PMID of each citing
article by reusing exactly the same PMID -> DOI -> Scopus ID cascade
that already exists in scopus_to_pmid.py (_resolve_one), downloads its
MeSH metadata from PubMed (fetch_pubmed_details), and classifies it as
A/C/H by reusing mesh_classification.py.

Every other citation that Scopus actually reports DOES get a row in
citations_details.csv, no matter how little is known about it:
- if it cannot be resolved to a PMID at all, it is kept with
  CitingCategory = "NOT_FOUND";
- if it has a PMID but no MeSH terms in PubMed, or its MeSH terms
  don't fall under any A/C/H branch, it is kept with
  CitingCategory = "UNCATEGORIZED".

Both cases are deliberately kept rather than discarded, because even
an unclassifiable citation still works as a "bridge" node: the next
BFS generation explores ITS OWN forward citations too (by ScopusID,
which does not require a PMID — see _run_bfs_for_root's frontier
construction), so a chain of otherwise-unclassifiable citations can
still lead to an H-category article several generations down the
line.

Outputs:
- citations_details.csv: one row per original -> citing relationship,
  with the full detail of the citation (ScopusID, PMID, MeSH, Category,
  etc.).
- papers_full.csv (updated in place): TD, TY, TC and SearchStatus
  columns filled in for each original article once its forward-citation
  search finishes.

Edge direction (consistent with the project's README): the citation
comes after the original, so the edge is original -> citing article
("the original is cited by the citing article").

Usage as a script (main.py's third mode):
    python -m scripts.main --forward-citations results/papers_full.csv \
        --api-key YOUR_API_KEY --output-dir results/
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union
from xml.etree import ElementTree as ET
from .data_io import read_csv_clean, to_csv_clean

import pandas as pd

from .mesh_classification import classify_mesh_codes, get_mesh_tree_numbers, load_mesh_indices
from .scopus_to_pmid import (
    ApiKeyPool, QuotaExceededError, _do_request, _pmid_from_title, fetch_pubmed_details, ESEARCH_URL,
)

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"

_NOT_FOUND_LABEL = "NOT_FOUND"
_UNCATEGORIZED_LABEL = "UNCATEGORIZED"

_LIST_SEP = "|"
_CHECKPOINT_EVERY = 10  # ORIGINAL articles processed between checkpoints

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_MAX_GENERATIONS = 4

DOI_ESEARCH_BATCH_SIZE = 40  # DOIs per OR query (a conservative URL-length limit)

def _norm_id(x):
    """
    Normalizes any ID-like value (ScopusID, PMID, etc.) to a canonical
    string, so it can be reliably used as a dictionary key or to compare
    two IDs with each other.

    Fixes the same type-coercion problem documented in data_io.py: an ID
    that starts out as text can end up reinterpreted as float at some
    point in the pipeline (e.g. when passing through a DataFrame without
    the dtype forced), and then shows up with a ",0" or ".0" suffix
    stuck to it. This function strips both suffixes if present.

    A null value (NaN/None, via pd.isna) is normalized to an empty
    string "" rather than to "nan" or similar, so it can be used
    directly in comparisons and as a key without having to check
    beforehand whether it is null at every call site.
    """
    if pd.isna(x):
        return ""
    return str(x).strip().replace(",0", "").replace(".0", "")

def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalizes a DOI for reliable comparison: lowercase, no
    whitespace, no URL/scheme prefix."""
    if not doi:
        return None
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip() or None


def _esearch_pmids_by_doi_batch(dois_batch: list[str], timeout: int = 30) -> list[str]:
    """
    A single esearch call with an OR of several DOIs at once. Returns
    the list of candidate PMIDs that match ANY of the DOIs in the
    batch — WITHOUT yet knowing which PMID corresponds to which DOI
    (that is reconciled afterward by comparing against the real DOI
    efetch returns, which has to be called anyway for the MeSH data).
    """
    if not dois_batch:
        return []
    term = " OR ".join(f"{doi}[DOI]" for doi in dois_batch)
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "term": term,
        "retmax": str(max(100, len(dois_batch) * 3)),
    }
    r = _do_request("GET", ESEARCH_URL, service="PubMed", params=params, timeout=timeout)
    if r.status_code != 200:
        return []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []
    return [el.text for el in root.findall(".//Id") if el.text]


def _batch_resolve_pmid_candidates_by_doi(
    dois: list[str],
    batch_size: int = DOI_ESEARCH_BATCH_SIZE,
    sleep_time: float = 0.2,
    verbose: bool = False,
) -> list[str]:
    """Resolves a list of DOIs to candidate PMIDs, in batches of
    `batch_size`, with one OR per batch. Returns the union of every
    candidate PMID found (still not reconciled with its DOI)."""
    unique_dois = list(dict.fromkeys(d for d in dois if d))
    if not unique_dois:
        return []
    candidates: list[str] = []
    n_batches = (len(unique_dois) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = unique_dois[b * batch_size:(b + 1) * batch_size]
        if verbose:
            print(f"      [esearch DOI batch {b + 1}/{n_batches}, {len(batch)} DOIs]")
        candidates.extend(_esearch_pmids_by_doi_batch(batch))
        time.sleep(sleep_time)
    return list(dict.fromkeys(candidates))

def _collect_generation_entries(
    frontier: list[dict],
    key_pool: ApiKeyPool,
    sleep_time: float,
    generation: int,
    debug: bool,
) -> list[dict]:
    """
    Walks EVERY node of the frontier and calls Scopus (REF) for each one.
    It does NOT resolve PMID yet (that is done in a batch,
    afterward, for the whole generation at once).
    Returns a flat list of entries, each with its parent_node_id.
    """
    all_entries: list[dict] = []
    for node in frontier:
        citing_entries = _fetch_all_citing_entries(node["ScopusID"], key_pool, sleep_time)
        filtered = [
            e for e in citing_entries
            if e["ScopusID"] and e["Year"] is not None and node["Year"] <= e["Year"]
        ]
        if debug:
            print(f"      [gen {generation}] Node {node['node_id']} (Scopus {node['ScopusID']}, "
                  f"year {node['Year']}): Scopus returned {len(filtered)} citations in total.")

        for e in filtered:
            all_entries.append({
                "parent_node_id": node["node_id"],
                "parent_pmid": node.get("pmid"),
                "parent_ScopusID": node["ScopusID"],
                "ScopusID": e["ScopusID"],
                "doi": e.get("doi"),
                "Title": e.get("Title"),
                "Year": e["Year"],
                "pmid": e.get("pmid"),
            })

    return all_entries

def _build_reuse_caches_from_prev(prev: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Builds the 3 reuse caches from ALL of the rows of a previous
    citations_details.csv: a citation's identity (its PMID, MeSH,
    category, DOI) is always reusable, avoiding repeated calls to
    Scopus/PubMed.

    Returns:
        (citations_cache, doi_pmid_cache, scopus_pmid_cache)
    """
    citations_cache: dict[str, dict] = {}
    doi_pmid_cache: dict[str, str] = {}
    scopus_pmid_cache: dict[str, str] = {}

    for _, prow in prev.iterrows():
        raw_ScopusID = prow.get("CitingScopusID")
        ScopusID = str(raw_ScopusID).strip() if pd.notna(raw_ScopusID) else ""
        pmid = prow.get("CitingPMID")
        pmid = str(pmid).strip() if pd.notna(pmid) and str(pmid).strip() else None

        # scopus_pmid_cache stores, for each ScopusID already seen, its
        # PMID if one was found, or "" if it was already attempted and
        # it has none (distinguishable from "not attempted yet", which
        # is the key's outright absence). This avoids repeating the
        # DOI/title search for a citation already known to have no
        # PMID.
        if ScopusID:
            scopus_pmid_cache[ScopusID] = pmid if pmid else ""

        # citations_cache stores, by PMID, the MeSH/category/DOI detail
        # already computed for that citation, so that if the same
        # article shows up again as a citation of ANOTHER original, it
        # does not need to be requested from PubMed or reclassified
        # again.
        if pmid and pmid not in citations_cache:
            citations_cache[pmid] = {
                "MeSHList": prow.get("CitingMeSHList"),
                "MeSHTerms": prow.get("CitingMeSHTerms"),
                "MeSHTreeNumbers": prow.get("CitingMeSHTreeNumbers"),
                "Category": prow.get("CitingCategory"),
                "DOI": prow.get("CitingDOI"),
            }
            # doi_pmid_cache lets a later citation be resolved to its
            # PMID directly from its DOI without calling esearch, if
            # that DOI already appeared in an earlier processed row.
            norm_doi = _normalize_doi(prow.get("CitingDOI"))
            if norm_doi:
                doi_pmid_cache[norm_doi] = pmid

    return citations_cache, doi_pmid_cache, scopus_pmid_cache

def _resolve_generation_entries(
    entries: list[dict],
    originals_by_pmid: dict,
    citations_cache: dict,
    doi_pmid_cache: dict,
    scopus_pmid_cache: dict,
    ui_index: dict,
    name_index: dict,
    pubmed_api_key: Optional[str],
    pubmed_sleep_time: float,
    debug: bool,
) -> list[dict]:
    """
    Resolves PMID + MeSH + category for ALL of the entries of one
    entire generation, minimizing calls to PubMed:

      1. Direct PMIDs (already present in the Scopus entry).
      2. PMIDs already known by DOI in doi_pmid_cache (no call needed).
      3. Batch (OR) search in esearch for the remaining DOIs.
      4. efetch in a SINGLE batch for every PMID collected so far
         (direct + DOI candidates): brings back MeSH and each one's
         real DOI, used to reconcile.
      5. DOI -> PMID reconciliation by comparing the normalized DOI.
      6. Title fallback (individually, one by one) for whatever still
         has no PMID.
      7. Second batched efetch for the new PMIDs resolved by title.
      8. Building the final rows. Nothing is discarded: no PMID ->
         NOT_FOUND; PMID but no A/C/H match -> UNCATEGORIZED.
    """
    # --- 0: PMID already known by ScopusID (from earlier runs) ---
    n_from_scopus_cache = 0
    for e in entries:
        if e["pmid"]:
            continue
        if e.get("_scopus_cache_tried"):
            continue  # already known by ScopusID to have no PMID, don't retry by DOI
        cached_pmid = scopus_pmid_cache.get(e["ScopusID"])
        if cached_pmid is not None:  # can be a real PMID, or "" marking "already tried, none found"
            if cached_pmid:
                e["pmid"] = cached_pmid
            e["_scopus_cache_tried"] = True
            n_from_scopus_cache += 1
    if debug and n_from_scopus_cache:
        print(f"      [gen] Resolved from scopus_pmid_cache (no call needed): {n_from_scopus_cache}")


    # --- 1-2: direct + already known by DOI in cache ---
    for e in entries:
        if e["pmid"]:
            continue
        norm_doi = _normalize_doi(e["doi"])
        if norm_doi and norm_doi in doi_pmid_cache:
            e["pmid"] = doi_pmid_cache[norm_doi]

    # --- 3: batch search for the remaining DOIs without a PMID ---
    dois_to_search = [
        _normalize_doi(e["doi"]) for e in entries
        if not e["pmid"] and e["doi"] and not e.get("_scopus_cache_tried")
    ]
    doi_candidates = _batch_resolve_pmid_candidates_by_doi(
        dois_to_search, sleep_time=pubmed_sleep_time, verbose=debug,
    )

    # --- 4: batched efetch of everything collected so far ---
    direct_pmids = [e["pmid"] for e in entries if e["pmid"]]
    pmids_to_fetch = list(dict.fromkeys(
        [p for p in direct_pmids if p not in originals_by_pmid and p not in citations_cache]
        + [p for p in doi_candidates if p not in originals_by_pmid and p not in citations_cache]
    ))
    if debug:
        print(f"      [gen] batched efetch: {len(pmids_to_fetch)} new PMIDs "
              f"({len(direct_pmids)} direct, {len(doi_candidates)} DOI candidates).")

    details_df = fetch_pubmed_details(pmids_to_fetch, sleep_time=pubmed_sleep_time, verbose=False)
    _store_pubmed_details_in_cache(details_df, citations_cache, doi_pmid_cache, ui_index, name_index)

    # --- 5: DOI -> PMID reconciliation ---
    for e in entries:
        if e["pmid"]:
            continue
        norm_doi = _normalize_doi(e["doi"])
        if norm_doi and norm_doi in doi_pmid_cache:
            e["pmid"] = doi_pmid_cache[norm_doi]

    # --- 6: title fallback, one at a time ---
    still_missing = [
        e for e in entries
        if not e["pmid"] and e.get("Title") and not e.get("_scopus_cache_tried")
    ]
    if debug and still_missing:
        print(f"      [gen] Title fallback: {len(still_missing)} citations without a PMID by DOI.")

    title_resolved_pmids = []
    for e in still_missing:
        pmid, _ = _pmid_from_title(e["Title"], scopus_doi=e.get("doi"), scopus_year=str(e.get("Year")) if e.get("Year") else None, pubmed_api_key=pubmed_api_key)

        time.sleep(pubmed_sleep_time)
        if pmid:
            e["pmid"] = pmid
            title_resolved_pmids.append(pmid)

    # --- 7: batched efetch of the PMIDs resolved by title ---
    new_title_pmids = [
        p for p in dict.fromkeys(title_resolved_pmids)
        if p not in citations_cache and p not in originals_by_pmid
    ]
    if new_title_pmids:
        title_details_df = fetch_pubmed_details(new_title_pmids, sleep_time=pubmed_sleep_time, verbose=False)
        _store_pubmed_details_in_cache(title_details_df, citations_cache, doi_pmid_cache, ui_index, name_index)

    # --- 8: build the final rows ---
    rows = []
    for e in entries:
        pmid = e["pmid"]
        node_id = pmid if pmid else f"SCOPUS:{e['ScopusID']}"
        reused = (originals_by_pmid.get(pmid) or citations_cache.get(pmid)) if pmid else None
        scopus_pmid_cache[e["ScopusID"]] = pmid if pmid else ""

        if reused is not None:
            mesh_list = reused.get("MeSHList")
            mesh_terms = reused.get("MeSHTerms")
            tree_numbers_str = reused.get("MeSHTreeNumbers")
            category_str = reused.get("Category") or _UNCATEGORIZED_LABEL
        elif pmid is None:
            mesh_list, mesh_terms, tree_numbers_str = None, None, None
            category_str = _NOT_FOUND_LABEL
        else:
            # Has a PMID but its detail could not be obtained (rare: a
            # retracted PMID, or efetch did not return it).
            mesh_list, mesh_terms, tree_numbers_str = None, None, None
            category_str = _UNCATEGORIZED_LABEL

        rows.append({
            "Generation": None,
            "ParentNodeID": e["parent_node_id"],
            "ParentPMID": e.get("parent_pmid"),
            "ParentScopusID": e.get("parent_ScopusID"),
            "CitingNodeID": node_id,
            "CitingScopusID": e["ScopusID"],
            "CitingDOI": e.get("doi"),
            "CitingPMID": pmid,
            "CitingYear": e["Year"],
            "CitingMeSHList": mesh_list,
            "CitingMeSHTerms": mesh_terms,
            "CitingMeSHTreeNumbers": tree_numbers_str,
            "CitingCategory": category_str,
            "PathToH": False,
        })

    return rows

def _store_pubmed_details_in_cache(
    details_df: pd.DataFrame,
    citations_cache: dict,
    doi_pmid_cache: dict,
    ui_index: dict,
    name_index: dict,
) -> None:
    """Classifies and caches (by PMID and by DOI) each row of a
    DataFrame returned by fetch_pubmed_details."""
    for pmid, detail in details_df.to_dict(orient="index").items():
        mesh_list = detail.get("MeSHList")
        mesh_terms = detail.get("MeSHTerms")
        doi = detail.get("DOI")

        category = classify_mesh_codes(mesh_list, ui_index, name_index=name_index, mesh_terms=mesh_terms)
        tree_numbers = get_mesh_tree_numbers(mesh_list, ui_index, name_index=name_index, mesh_terms=mesh_terms)
        category_str = _category_to_str(category) or _UNCATEGORIZED_LABEL
        tree_numbers_str = _LIST_SEP.join(tree_numbers) if tree_numbers else None

        citations_cache[pmid] = {
            "MeSHList": mesh_list,
            "MeSHTerms": mesh_terms,
            "MeSHTreeNumbers": tree_numbers_str,
            "Category": category_str,
            "DOI": doi,
        }
        norm_doi = _normalize_doi(doi)
        if norm_doi:
            doi_pmid_cache[norm_doi] = pmid

def _run_bfs_for_root(
    root_row: pd.Series,
    key_pool: ApiKeyPool,
    pubmed_api_key: Optional[str],
    ui_index: dict,
    name_index: dict,
    originals_by_pmid: dict,
    citations_cache: dict,
    doi_pmid_cache: dict,
    scopus_pmid_cache: dict,
    sleep_time: float,
    pubmed_sleep_time: float,
    debug: bool = True,
) -> tuple[list[dict], float, float, float, str]:
    """
    BFS by COMPLETE generations: in each generation, ALL citations from
    ALL open branches are collected and resolved (in batches against
    PubMed) before deciding whether there is an H. If several H's show
    up in the same generation (in different branches), the one with the
    oldest year is chosen. If there is no H at all, the next generation
    is built from ALL of this generation's citations (with or without a
    PMID, UNCATEGORIZED or NOT_FOUND included) — nothing is dropped from
    the tree.
    """
    root_ScopusID = str(root_row["ScopusID"]).strip()
    root_pmid = str(root_row["PMID"]).strip()
    root_year = int(root_row["Year"])
    root_pmid_real = root_pmid if root_pmid and root_pmid.lower() != "nan" else None
    root_node_id = root_pmid_real or f"SCOPUS:{root_ScopusID}"
    if debug:
        print(f"  [BFS] Root {root_ScopusID} (PMID {root_pmid}, year {root_year}, "
              f"Category={root_row.get('Category')})")

    visited_ids = {root_node_id}
    frontier = [{
        "node_id": root_node_id, "pmid": root_pmid_real,
        "ScopusID": root_ScopusID, "Year": root_year,
    }]

    all_rows: list[dict] = []
    rows_by_id: dict[str, dict] = {}
    h_row: Optional[dict] = None

    for generation in range(1, _MAX_GENERATIONS + 1):
        if not frontier:
            if debug:
                print(f"  [BFS] Generation {generation}: empty frontier, stopping BFS for this root.")
            break

        if debug:
            print(f"  [BFS] === Generation {generation}: exploring {len(frontier)} node(s) ===")

        entries = _collect_generation_entries(frontier, key_pool, sleep_time, generation, debug)
        generation_rows = _resolve_generation_entries(
            entries, originals_by_pmid, citations_cache, doi_pmid_cache, scopus_pmid_cache,
            ui_index, name_index, pubmed_api_key, pubmed_sleep_time, debug=debug,
        )
        for row in generation_rows:
            row["Generation"] = generation

        for row in generation_rows:
            rows_by_id.setdefault(row["CitingNodeID"], row)
        all_rows.extend(generation_rows)

        if debug:
            cats_count: dict[str, int] = {}
            for r in generation_rows:
                cats_count[r["CitingCategory"]] = cats_count.get(r["CitingCategory"], 0) + 1
            print(f"  [BFS] Generation {generation} complete: {len(generation_rows)} citations total. "
                  f"Breakdown by category: {cats_count}")

        h_candidates = [r for r in generation_rows if r["CitingCategory"] == "H"]
        if h_candidates:
            h_row = min(h_candidates, key=lambda r: r["CitingYear"])
            if debug:
                print(f"  [BFS] Oldest H in generation {generation}: "
                      f"PMID {h_row['CitingPMID']}, year {h_row['CitingYear']}.")
            break

        next_frontier = []
        for row in sorted(generation_rows, key=lambda r: r["CitingYear"]):
            node_id = row["CitingNodeID"]
            if node_id in visited_ids:
                continue
            visited_ids.add(node_id)
            next_frontier.append({
                "node_id": node_id, "pmid": row["CitingPMID"],
                "ScopusID": row["CitingScopusID"], "Year": row["CitingYear"],
            })
        frontier = next_frontier
        if debug:
            print(f"  [BFS] Frontier for generation {generation + 1}: {len(next_frontier)} node(s).")

    if h_row is not None:
        current = h_row
        current["PathToH"] = True
        while current["ParentNodeID"] != root_node_id:
            parent_row = rows_by_id.get(current["ParentNodeID"])
            if parent_row is None:
                break
            parent_row["PathToH"] = True
            current = parent_row
        td = h_row["Generation"]
        ty = h_row["CitingYear"] - root_year
        tc = 1 / td
        search_status = "PROCESSED_REACHED_H"
    elif not all_rows:
        td, ty, tc = float("inf"), float("inf"), 0.0
        search_status = "PROCESSED_NO_CITATIONS"
    else:
        td, ty, tc = float("inf"), float("inf"), 0.0
        search_status = "PROCESSED_NOT_REACHED_H"

    if debug:
        print(f"  [BFS] Result for root {root_ScopusID}: TD={td}, TY={ty}, TC={tc:.4f}, "
              f"SearchStatus={search_status}, total rows generated={len(all_rows)}.")

    for row in all_rows:
        row["ScopusID"] = root_ScopusID
        row["PMID"] = root_pmid

    return all_rows, td, ty, tc, search_status

# =====================================
# SCOPUS SEARCH (REF(ScopusID)) -> list of citing entries
# =====================================
def _parse_citing_entry(entry: ET.Element) -> dict:
    """Extracts ScopusID, doi, Title and Year (int or None) from an
    <entry> of the Scopus Search API."""
    identifier = entry.findtext("dc:identifier", namespaces=_NS)
    ScopusID = identifier.split(":")[-1].strip() if identifier else None

    doi = entry.findtext("prism:doi", namespaces=_NS)
    Title = entry.findtext("dc:Title", namespaces=_NS)
    # The PMID, whenever Scopus already includes it directly in the
    # response, travels in a <pubmed-id> element under the "atom"
    # namespace (unlike the rest of this function's fields, which use
    # "dc" or "prism"); that is why that specific namespace is used
    # here.
    pmid = entry.findtext("atom:pubmed-id", namespaces=_NS)
    pmid = pmid.strip() if pmid else None

    cover_date = entry.findtext("prism:coverDate", namespaces=_NS)
    year = int(cover_date[:4]) if cover_date and cover_date[:4].isdigit() else None

    return {"ScopusID": ScopusID, "doi": doi, "pmid": pmid, "Title": Title, "Year": year}

def _fetch_all_citing_entries(
    ScopusID: str,
    key_pool: ApiKeyPool,
    sleep_time: float = 0.2,
    page_size: int = 25,
    timeout: int = 30,
    debug: bool = False,
) -> list[dict]:
    """
    Returns ALL of the entries (paginating) of the REF(ScopusID) search
    in the Scopus Search API: the articles that cite `ScopusID`.
    """
    entries: list[dict] = []
    start = 0
    total_results: Optional[int] = None

    while total_results is None or start < total_results:
        params = {
            "query": f"REF({ScopusID})",
            "count": page_size,
            "start": start,
        }
        while True:
            try:
                headers = {**key_pool.headers(), "Accept": "application/atom+xml"}
                r = _do_request(
                    "GET", SCOPUS_SEARCH_URL, service="Scopus",
                    headers=headers, params=params, timeout=timeout,
                )
                break
            except QuotaExceededError:
                if not key_pool.advance():
                    raise

        if r.status_code != 200:
            if debug:
                print(f"      [REF] ScopusID={ScopusID} -> HTTP {r.status_code}: {r.text[:250]}")
            break

        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as exc:
            if debug:
                print(f"      [REF] ScopusID={ScopusID} -> invalid XML ({exc}): {r.text[:250]}")
            break

        if total_results is None:
            total_elem = root.find("opensearch:totalResults", _NS)
            total_results = int(total_elem.text) if total_elem is not None and total_elem.text else 0

        page_entries = root.findall("atom:entry", _NS)
        if not page_entries:
            break

        entries.extend(_parse_citing_entry(e) for e in page_entries)

        start += page_size
        time.sleep(sleep_time)

    return entries


def _category_to_str(categories: frozenset) -> str:
    """Same as _category_to_str in main.py. It is duplicated (rather
    than imported) to avoid a circular import: main.py imports this
    module to expose the CLI's third mode."""
    order = {"A": 0, "C": 1, "H": 2}
    if not categories:
        return ""
    return "".join(sorted(categories, key=lambda c: order.get(c, 99)))

def _reconstruct_root_metrics_from_detail(
    ScopusID: str,
    root_year,
    detail_rows: list[dict],
) -> tuple:
    """
    Reconstructs TD/TY/TC for an original article already present in
    citations_details.csv (ScopusID in done_original_ids) from the rows
    already saved for that ScopusID, WITHOUT calling Scopus or PubMed
    again. Used when papers_full.csv lost those values (e.g. from an
    incomplete save on an earlier run) but citations_details.csv still
    has them.

    - Looks for the row marked PathToH=True with CitingCategory == "H":
      TD = its Generation, TY = its CitingYear - root_year, TC = 1/TD.
    - If there is no such row, the exploration is assumed to have
      finished without reaching H: TD/TY = inf, TC = 0.0.
    """
    own_rows = [r for r in detail_rows if str(r.get("ScopusID")).strip() == ScopusID]

    h_row = next(
        (r for r in own_rows if r.get("CitingCategory") == "H" and r.get("PathToH")),
        None,
    )

    if h_row is not None:
        td = int(float(h_row["Generation"]))
        ty = int(float(h_row["CitingYear"])) - int(root_year)
        tc = 1 / td
        search_status = "PROCESSED_REACHED_H"
    elif not own_rows:
        td, ty, tc = float("inf"), float("inf"), 0.0
        search_status = "PROCESSED_NO_CITATIONS"
    else:
        td, ty, tc = float("inf"), float("inf"), 0.0
        search_status = "PROCESSED_NOT_REACHED_H"

    return td, ty, tc, search_status

def compute_forward_citations(
    papers_full_csv: Union[str, Path],
    mesh_descriptors_xml: Union[str, Path],
    output_dir: Union[str, Path],
    api_keys: Union[str, list, tuple],
    pubmed_api_key: Optional[str] = None,
    sleep_time: float = 0.2,
    pubmed_sleep_time: Optional[float] = None,
    resume: bool = True,
    verbose: bool = True,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Computes the forward citations of every valid article in
    `papers_full_csv` (Category not in {NOT_FOUND, UNCATEGORIZED}),
    updates its TD/TY/TC/SearchStatus columns in the CSV itself, and
    generates citations_details.csv with the full detail of every
    citation.

    Quota handling (HTTP 429): same as scopus_to_pmid.py — it stops
    immediately with no retry, saving the progress made so far to both
    CSVs.

    Args:
        papers_full_csv: path to a papers_full.csv that already has
            PMID/Category/x/y/TI computed (stages 1-2 of the pipeline,
            see main.py). It is OVERWRITTEN (TD/TY/TC/SearchStatus
            columns).
        mesh_descriptors_xml: path to the NLM's desc*.xml.
        output_dir: folder where citations_details.csv is saved.
        api_key: Elsevier/Scopus API key.
        pubmed_api_key: NCBI/PubMed E-utilities API key. It is OPTIONAL:
            without it, E-utilities still works the same, just limited
            to 3 requests/second per IP instead of 10 with a key.
        sleep_time: pause between requests to Scopus (s).
        pubmed_sleep_time: pause between efetch batches (s). If not
            given (None), it is computed on its own depending on
            whether pubmed_api_key is set: 0.3 s with a key (~10 req/s),
            0.5 s without one (~2 req/s, with margin under NCBI's real
            3 req/s limit).
        resume: if True, reuses an existing citations_details.csv and
            skips original articles already processed.
        verbose: if True, prints progress to the console.

    Returns:
        DataFrame with the final content of citations_details.csv.
    """
    if pubmed_sleep_time is None:
        pubmed_sleep_time = 0.3 if pubmed_api_key else 0.5
    key_pool = ApiKeyPool(api_keys)

    papers_full_path = Path(papers_full_csv)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    detail_csv_path = output_dir_path / "citations_details.csv"

    df = read_csv_clean(papers_full_path)
    df["PMID"] = df["PMID"].astype(str).str.replace(r"\.0$|,0$", "", regex=True).replace("nan", None)
    df["TD"] = df.get("TD", pd.Series(dtype="object")).astype("object")
    df["TY"] = df.get("TY", pd.Series(dtype="object")).astype("object")
    df["TC"] = df.get("TC", pd.Series(dtype="float64"))
    df["SearchStatus"] = df.get("SearchStatus", pd.Series(dtype="object"))
    df["SearchStatus"] = df["SearchStatus"].fillna("NOT_PROCESSED")

    # PMID -> already-computed-data index for the ORIGINAL articles, to
    # avoid calling PubMed again when a citation turns out to itself be
    # one of the 58k articles we already have processed.
    originals_by_pmid: dict[str, dict] = {}
    for _, orow in df.dropna(subset=["PMID"]).iterrows():
        pmid = str(orow["PMID"]).strip()
        if pmid:
            originals_by_pmid[pmid] = {
                "MeSHList": orow.get("MeSHList"),
                "MeSHTerms": orow.get("MeSHTerms"),
                "MeSHTreeNumbers": orow.get("MeSHTreeNumbers"),
                "Category": orow.get("Category"),
            }

    print(f"Computing forward citations from '{papers_full_csv}'")
    citations_cache: dict[str, dict] = {}
    doi_pmid_cache: dict[str, str] = {}
    scopus_pmid_cache: dict[str, str] = {}
    for pmid, data in originals_by_pmid.items():
        norm_doi = _normalize_doi(data.get("DOI"))
        if norm_doi:
            doi_pmid_cache[norm_doi] = pmid
    candidates = df

    ui_index, name_index = load_mesh_indices(mesh_descriptors_xml)

    all_rows: list[dict] = []
    done_original_ids: set[str] = set()
    row_index = {}
    if resume and detail_csv_path.exists():
        prev = read_csv_clean(detail_csv_path)
        all_rows = prev.to_dict("records")
        row_index = {
            (
                _norm_id(r["ScopusID"]),
                int(r["Generation"]),
                # _norm_id always returns a real str (never pd.NA), so
                # this "or" between two already-cleaned columns is safe.
                _norm_id(r.get("ParentNodeID")) or _norm_id(r.get("ParentPMID")),
                _norm_id(r["CitingNodeID"]),
            ): i
            for i, r in enumerate(all_rows)
        }

        # A root is considered "done" (skipped) if it already has rows
        # in citations_details.csv: it is reused as is, with no
        # reprocessing.
        done_original_ids = set(prev["ScopusID"].astype(str))

        prev_citations_cache, prev_doi_pmid_cache, prev_scopus_pmid_cache = _build_reuse_caches_from_prev(prev)
        citations_cache.update(prev_citations_cache)
        doi_pmid_cache.update(prev_doi_pmid_cache)
        scopus_pmid_cache.update(prev_scopus_pmid_cache)

        if verbose and done_original_ids:
            print(f"Resuming: {len(done_original_ids)} original articles already processed, skipping them.")
        if verbose and citations_cache:
            print(f"Citation cache recovered: {len(citations_cache)} PMIDs, "
                  f"{len(doi_pmid_cache)} DOIs, {len(scopus_pmid_cache)} ScopusIDs already computed.")

        n_reconstructed = 0
        for ScopusID in done_original_ids:
            match = df.index[df["ScopusID"] == ScopusID]
            if len(match) == 0:
                continue
            needs_reconstruction = [idx for idx in match if pd.isna(df.at[idx, "TD"])]
            if not needs_reconstruction:
                continue

            root_year = df.at[needs_reconstruction[0], "Year"]
            td, ty, tc, search_status = _reconstruct_root_metrics_from_detail(
                ScopusID, root_year, all_rows,
            )
            for idx in needs_reconstruction:
                df.at[idx, "TD"] = td
                df.at[idx, "TY"] = ty
                df.at[idx, "TC"] = tc
                df.at[idx, "SearchStatus"] = search_status
            n_reconstructed += len(needs_reconstruction)

        if verbose and n_reconstructed:
            print(f"Metrics reconstructed from citations_details.csv: {n_reconstructed} original articles.")
            to_csv_clean(df, papers_full_path)

    def _save_progress() -> None:
        """
        Flushes the state accumulated so far to disk: every citation
        row collected so far (`all_rows`, one per original->citing
        relationship) into citations_details.csv, and the `df`
        DataFrame of papers_full (with its TD/TY/TC/SearchStatus
        columns already updated for the original articles processed so
        far) into its own CSV.

        Called both at the periodic checkpoints during the main loop
        and when the function finishes normally, and also if some API's
        quota runs out mid-run — in that last case it is what lets the
        work be resumed later without losing the work already
        achieved (see the `resume` parameter).
        """
        allr_rows_df = pd.DataFrame(all_rows)
        to_csv_clean(allr_rows_df, detail_csv_path)
        to_csv_clean(df, papers_full_path)

    n = len(candidates)
    processed_since_checkpoint = 0

    for pos, (i, row) in enumerate(candidates.iterrows(), start=1):
        # A row with no ScopusID and/or no Year cannot be processed:
        # ScopusID is needed to identify the article and look up its
        # citations, and Year is needed to compute TY (root_year, in
        # _run_bfs_for_root). Rather than letting it crash further down
        # with an unclear error (e.g. int(NaN) when converting the
        # year), it is detected here and this row is skipped, leaving a
        # record in SearchStatus of why it ended up without TD/TY/TC.
        # This is exactly the situation metrics.py already expects and
        # excludes under the "SKIPPED_INVALID_ROW" status.
        raw_ScopusID = row.get("ScopusID")
        raw_year = row.get("Year")
        row_is_invalid = (
            pd.isna(raw_ScopusID)
            or not str(raw_ScopusID).strip()
            or str(raw_ScopusID).strip().lower() == "nan"
            or pd.isna(raw_year)
        )
        if row_is_invalid:
            if verbose:
                print(f"[{pos}/{n}] (row without a valid ScopusID and/or Year, skipping)")
            df.at[i, "SearchStatus"] = "SKIPPED_INVALID_ROW"
            continue

        original_ScopusID = str(raw_ScopusID).strip()
        if original_ScopusID in done_original_ids:
            if verbose:
                print(f"[{pos}/{n}] {original_ScopusID} (reused)")
            continue

        if verbose:
            print(f"[{pos}/{n}] {original_ScopusID}")

        try:
            detail_rows, td, ty, tc, search_status = _run_bfs_for_root(
                row, key_pool, pubmed_api_key, ui_index, name_index, originals_by_pmid, citations_cache,
                doi_pmid_cache, scopus_pmid_cache,
                sleep_time, pubmed_sleep_time, debug=debug,
            )
        except QuotaExceededError as exc:
            _save_progress()
            print()
            print(f"[STOPPED] {exc.service} quota exhausted (HTTP 429).")
            print(f"         Reset reported by the API: {exc.reset_human()}")
            print(f"         Progress saved to '{detail_csv_path}' and '{papers_full_path}'.")
            return pd.DataFrame(all_rows)

        for new_row in detail_rows:
            key = (
                _norm_id(new_row["ScopusID"]),
                int(new_row["Generation"]),
                _norm_id(new_row["ParentNodeID"]),
                _norm_id(new_row["CitingNodeID"]),
            )

            if key in row_index:
                all_rows[row_index[key]] = new_row
            else:
                row_index[key] = len(all_rows)
                all_rows.append(new_row)
        same_scopus_id_mask = df["ScopusID"] == original_ScopusID
        df.loc[same_scopus_id_mask, "TD"] = td
        df.loc[same_scopus_id_mask, "TY"] = ty
        df.loc[same_scopus_id_mask, "TC"] = tc
        df.loc[same_scopus_id_mask, "SearchStatus"] = search_status

        done_original_ids.add(original_ScopusID)
        processed_since_checkpoint += 1

        if processed_since_checkpoint >= _CHECKPOINT_EVERY:
            _save_progress()
            processed_since_checkpoint = 0
            if verbose:
                print(f"    (checkpoint saved: {pos}/{n} original articles)")

    _save_progress()

    if verbose:
        print()
        print("Forward citations computation DONE.")
        print(f"Relationships generated: {len(all_rows)}")
        print(f"citations_details.csv saved to: {detail_csv_path.resolve()}")

    return pd.DataFrame(all_rows)