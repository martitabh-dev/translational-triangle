"""
mesh_classification.py

This is the first half of pipeline STAGE 2 (the second half is
geometry.py): it classifies each publication into one or more of the
three translational-triangle categories — Animal (A), Cell (C), Human
(H) — based on the MeSH descriptors already resolved for it in stage 1
(scopus_to_pmid.py: the MeSHList/MeSHTerms columns).

Mapping (per the Weber framework this project is based on):
- H (Human): the MeSH subtree hanging from HUMAN_ROOTS (see below, the
  "Humans" branch within Eukaryota) and the M01 (Persons) subtree.
- A (Animal): the entire B01 (Eukaryota) subtree EXCEPT the Human
  branch above — that is, any eukaryotic organism that is not human
  counts as Animal for the purposes of this framework (this includes,
  for example, plants and fungi, not only animals in the strict sense:
  the official MeSH tree's definition is inherited as is).
- C (Cell): the A11 (Cells), B03 (Bacteria), B04 (Viruses), G02.111.570
  (Molecular Structures) and G02.149 (Chemical Processes) subtrees —
  together, everything that represents research at the
  cellular/molecular/microbiological level, short of a complete
  organism.

The same article can fall into several categories at once (e.g. a
study with an animal model AND human patients is classified as
{"A", "H"} simultaneously, because it will have MeSH descriptors from
both branches). There is no priority or exclusion between categories:
the classification result is always a set (frozenset), never a single
label.

Classification is based on each MeSH descriptor's TREE NUMBERS
(TreeNumber), not directly on its UI code, because it is the position
within the MeSH hierarchical tree that determines which branch
(Animal, Cell, Human) a concept belongs to — the UI code itself (e.g.
"D000818") does not by itself indicate that hierarchy. That is why a
prior index is needed, built from the official, complete MeSH
descriptor file (desc*.xml, downloadable from
https://www.nlm.nih.gov/mesh/xmlmesh.html), which is what links each
descriptor to its TreeNumber (or TreeNumbers: the same descriptor can
appear in several tree branches at once).

Per-descriptor lookup strategy (in this order):
1. Look up the UI code (MeSHList column, e.g. "D000818") directly in
   the UI -> TreeNumbers index, built from <DescriptorUI>.
2. If the UI does not appear in the index (e.g. an obsolete descriptor,
   or a version mismatch of the MeSH thesaurus between when the CSV was
   generated and the desc*.xml being used), it falls back to looking up
   the descriptor's NAME (MeSHTerms column, e.g. "Animals") in a second
   index, Name -> TreeNumbers, built from <DescriptorName><String>.
   This works because MeSHList and MeSHTerms were generated in
   parallel, position by position, in scopus_to_pmid.py
   (_parse_pubmed_article): the term at position i of MeSHTerms is the
   readable name of the code at position i of MeSHList.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import pandas as pd

# MeSH tree roots that define each translational-triangle category. A
# descriptor "belongs" to a category if its TreeNumber is exactly one
# of these roots, or hangs from it (see _is_under, below, for the exact
# "hangs from" criterion).
#
# HUMAN_ROOTS combines two independent branches of the MeSH tree: the
# specific "Humans" branch within the organism hierarchy (B01...) and
# the M01 (Persons) subtree, which groups person-related concepts from
# a different angle of the thesaurus (roles, population groups, etc.),
# not just the biological organism itself.
HUMAN_ROOTS: FrozenSet[str] = frozenset({
    "B01.050.150.900.649.313.988.400.112.400.400",
    "M01",
})
# CELL_ROOTS groups the branches that represent research at the
# cellular, molecular or microbiological level: A11 (Cells, the subtree
# of cell types), B03/B04 (Bacteria/Viruses, microorganisms often
# studied with the same techniques as pure cell work), and
# G02.111.570/G02.149 (Molecular Structures/Chemical Processes,
# molecular-level processes and structures).
CELL_ROOTS: FrozenSet[str] = frozenset({
    "A11",
    "B03",
    "B04",
    "G02.111.570",
    "G02.149",
})
# Root of the Animal category: the WHOLE B01 (Eukaryota) subtree counts
# as Animal, except for the Human branch explicitly excluded below
# (_ANIMAL_EXCLUDED_SUBTREE). Being the root of the entire "Eukaryota",
# this category in practice covers any non-human eukaryotic organism
# (it is not limited to animals in the colloquial sense of the word):
# that is how the official MeSH tree defines it, and this pipeline
# inherits that definition as is.
ANIMAL_ROOT: str = "B01"

# The Human branch within B01 that must be EXCLUDED when checking
# Animal, so that a Humans descriptor does not ALSO count as Animal (it
# already counts as Human via HUMAN_ROOTS). It must match exactly the
# first TreeNumber of HUMAN_ROOTS: it is the same branch, repeated here
# as its own constant because _classify_tree_number needs it as a
# single value (not as part of a set) for the Animal check's `and not`.
_ANIMAL_EXCLUDED_SUBTREE: str = "B01.050.150.900.649.313.988.400.112.400.400"

# Separator used in the MeSHList / MeSHTerms columns (see
# scopus_to_pmid.py: _LIST_SEP).
_SEP = "|"


def build_mesh_indices(
    descriptor_xml_path: Union[str, Path],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Parses the OFFICIAL, COMPLETE MeSH descriptor file (desc*.xml, the
    full thesaurus published by the NLM, which contains EVERY MeSH
    descriptor that exists, not just the ones that appear in this
    dataset) and builds TWO indices in a single pass over the file:

    - ui_index:   { DescriptorUI: [TreeNumber, ...] }
    - name_index: { DescriptorName (exact text): [TreeNumber, ...] }

    The same descriptor can have several TreeNumbers (it can appear in
    more than one branch of the MeSH tree at once), which is why each
    entry's value is a LIST, not a single TreeNumber.

    The second index (name_index) serves as a fallback for when a
    MeSHList UI code does not appear in ui_index (e.g. due to a
    thesaurus version mismatch between when the CSV was generated and
    the desc*.xml being used now), allowing the same descriptor to then
    be located by its readable name (MeSHTerms column) instead of by
    its code.

    Implementation detail: `ET.iterparse` is used (incremental,
    node-by-node parsing) instead of `ET.parse` (which would load the
    entire XML tree into memory at once) because the full desc*.xml
    file can be several hundred MB. After processing each
    <DescriptorRecord>, `elem.clear()` is called to free that node (and
    its children) from memory as soon as it is no longer needed,
    preventing memory usage from growing unbounded as thousands of
    descriptors are read.

    Args:
        descriptor_xml_path: path to the NLM's official desc*.xml file
            (e.g. "dataset/desc2026.xml").

    Returns:
        Tuple (ui_index, name_index), both dictionaries already
        complete with EVERY descriptor found in the file (not only the
        ones relevant to the translational triangle: category filtering
        is done later, in _classify_tree_number, not here). A
        <DescriptorRecord> with no TreeNumber at all (a rare but
        possible case for some thesaurus record types) is not added to
        either index.
    """
    ui_index: Dict[str, List[str]] = {}
    name_index: Dict[str, List[str]] = {}
    context = ET.iterparse(str(descriptor_xml_path), events=("end",))

    for _, elem in context:
        # iterparse walks EVERY node of the XML, not just
        # <DescriptorRecord>; any other tag is skipped (for example, the
        # inner nodes of each record, which iterparse also visits as
        # they get closed) so that each descriptor is processed exactly
        # once, already complete.
        if elem.tag != "DescriptorRecord":
            continue

        ui_elem = elem.find("DescriptorUI")
        ui = ui_elem.text.strip() if ui_elem is not None and ui_elem.text else None

        name_elem = elem.find("DescriptorName/String")
        name = name_elem.text.strip() if name_elem is not None and name_elem.text else None

        tree_numbers = [
            tn.text.strip()
            for tn in elem.findall("./TreeNumberList/TreeNumber")
            if tn.text
        ]

        if tree_numbers:
            # The descriptor is only registered in the indices if it has
            # AT LEAST one TreeNumber: without a TreeNumber there is no
            # way to know which branch of the tree it falls under, so it
            # would contribute nothing to classification even if saved.
            if ui:
                ui_index[ui] = tree_numbers
            if name:
                name_index[name] = tree_numbers

        elem.clear()  # frees memory for the already-processed node

    return ui_index, name_index


@functools.lru_cache(maxsize=1)
def _cached_mesh_indices(
    descriptor_xml_path: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Caches build_mesh_indices' result in memory, so the full XML does
    not have to be re-parsed (an expensive operation, both in time and
    memory) every time load_mesh_indices is called with the same path.
    `maxsize=1` is intentional: this pipeline only ever works with one
    MeSH descriptor file at a time, so there is no need to cache more
    than one distinct path simultaneously; if called with a different
    path, the previous cache entry is simply discarded and recomputed
    for the new one."""
    return build_mesh_indices(descriptor_xml_path)


def load_mesh_indices(
    descriptor_xml_path: Union[str, Path],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Public entry point to obtain the two MeSH indices (ui_index,
    name_index), with caching in between (see _cached_mesh_indices):
    the first call parses the full XML, and any later call with the
    SAME path returns the already-computed result without touching the
    file again. `descriptor_xml_path` is converted to str before being
    passed to the cached function because functools.lru_cache needs its
    arguments to be stably "hashable", and a Path object and the
    equivalent string must be treated as the same cache key.

    This is the function main.py uses (in _classify_papers_full) to
    obtain the indices before classifying an entire DataFrame.

    Args:
        descriptor_xml_path: path to the NLM's official desc*.xml file.

    Returns:
        Tuple (ui_index, name_index), same as build_mesh_indices.
    """
    return _cached_mesh_indices(str(descriptor_xml_path))


def _is_under(tree_number: str, root: str) -> bool:
    """Checks whether a specific TreeNumber belongs to a branch of the
    MeSH tree, that branch being exactly `root` or any descendant of it.

    MeSH TreeNumbers are strings of numeric segments separated by dots
    (e.g. "B01.050.150.900..."), where each dot adds one more level of
    depth in the tree: that is why "hanging from root" is checked with
    `tree_number.startswith(root + ".")` — the dot is added explicitly
    so as not to confuse, for example, "B01" with "B0199" (which would
    start with "B01" as plain text, but does NOT hang from the tree's
    B01 branch).

    Args:
        tree_number: the TreeNumber to check (e.g. "B01.050.150").
        root: the branch root being checked against (e.g. "B01").

    Returns:
        True if tree_number is exactly root, or hangs from that branch
        (starts with "root."); False in any other case.
    """
    return tree_number == root or tree_number.startswith(root + ".")


def _classify_tree_number(tree_number: str) -> Dict[str, bool]:
    """Determines, for a SINGLE TreeNumber, which of the three
    categories (A/C/H) it belongs to, checking each one independently —
    a single TreeNumber could, in theory, trigger more than one
    category at once if the definitions overlapped, although with the
    roots defined in this file that does not actually happen in
    practice thanks to the explicit Animal/Human exclusion.

    - is_human: True if it hangs from any of the HUMAN_ROOTS roots.
    - is_cell: True if it hangs from any of the CELL_ROOTS roots.
    - is_animal: True if it hangs from ANIMAL_ROOT (B01) BUT does not
      hang from _ANIMAL_EXCLUDED_SUBTREE (the Human branch within B01)
      — this way a TreeNumber from the Human branch never also counts
      as Animal.

    Args:
        tree_number: a single MeSH TreeNumber to classify.

    Returns:
        Dictionary {"A": bool, "C": bool, "H": bool} with the result of
        the three checks for this specific TreeNumber.
    """
    is_human = any(_is_under(tree_number, root) for root in HUMAN_ROOTS)
    is_cell = any(_is_under(tree_number, root) for root in CELL_ROOTS)
    # Animal = B01 (Eukaryota) except the specific Human branch.
    is_animal = _is_under(tree_number, ANIMAL_ROOT) and not _is_under(
        tree_number, _ANIMAL_EXCLUDED_SUBTREE
    )
    return {"A": is_animal, "C": is_cell, "H": is_human}


def _as_list(value: Optional[Union[str, List[str]]], sep: str) -> List[str]:
    """Normalizes a MeSHList/MeSHTerms value, which can arrive in
    several different forms depending on where it comes from (directly
    from a DataFrame freshly read from a CSV, or already as a Python
    list if some caller built it by hand), into a single common form: a
    list of clean strings (no extra whitespace), the list every other
    function in this module works with.

    Cases handled:
      - `value` is None, or is a float NaN (this is how pandas
        represents an empty cell when reading a CSV): an empty list is
        returned.
      - `value` is a string: it is split on `sep` (normally "|", see
        _SEP) and each element is cleaned, discarding any that end up
        empty after `strip()` (e.g. from a duplicated separator
        "A||B").
      - `value` is already a list (or another iterable): each element
        that is a string is cleaned, leaving any non-string element as
        is (although in practice this should not happen:
        MeSHList/MeSHTerms are always text), and "falsy" elements
        (None, empty string...) are dropped.

    Args:
        value: the MeSHList or MeSHTerms cell value to normalize.
        sep: separator to use if `value` is a string.

    Returns:
        List of clean strings (may be empty).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(sep) if v.strip()]
    return [v.strip() if isinstance(v, str) else v for v in value if v]


def _lookup_tree_numbers(
    codes: List[str],
    ui_index: Dict[str, List[str]],
    name_index: Optional[Dict[str, List[str]]] = None,
    terms: Optional[List[str]] = None,
) -> List[str]:
    """
    Looks up the TreeNumbers for each UI code in `codes`, applying to
    each one the two-step lookup strategy described at the top of this
    file: first by UI in `ui_index`; if it does not appear there, and
    `name_index` and the corresponding name in `terms` (at the SAME
    position as the UI within `codes`) are available, that name is
    looked up in `name_index` as a fallback.

    This is the lookup logic shared by `classify_mesh_codes` (which
    uses it to decide which A/C/H categories the article belongs to)
    and `get_mesh_tree_numbers` (which directly exposes the list of
    TreeNumbers found, to be saved in the results CSV for
    traceability).

    Args:
        codes: list of MeSH UI codes (e.g. ["D000818", "D006801"]),
            already normalized with _as_list.
        ui_index: UI -> TreeNumbers index (from build_mesh_indices/
            load_mesh_indices).
        name_index: Name -> TreeNumbers index, used as a fallback. If
            omitted (None), there simply is no fallback: a UI that does
            not appear in ui_index is just skipped.
        terms: list of readable names, at the SAME position as `codes`
            (terms[i] is the name of code codes[i]). Used only when the
            name fallback is needed.

    Returns:
        A FLAT list with every TreeNumber found for every code in
        `codes`, preserving duplicates (if two different descriptors of
        the same article hang from the same TreeNumber, or if the same
        descriptor has several TreeNumbers, all of them appear in the
        output list without being deduplicated here — deduplication,
        when needed, is done by the caller).
    """
    terms = terms or []
    all_tree_numbers: List[str] = []

    for i, ui in enumerate(codes):
        if not ui:
            continue

        tree_numbers = ui_index.get(ui, [])

        if not tree_numbers and name_index is not None and i < len(terms):
            # Name fallback: only attempted if the UI lookup gave no
            # result (tree_numbers empty) AND a name is actually
            # available at that same position `i` within `terms` (in
            # case terms were shorter than codes, which should not
            # happen under normal conditions, but `i < len(terms)` is
            # checked as a safety measure to avoid raising an
            # IndexError).
            term = terms[i]
            if term:
                tree_numbers = name_index.get(term, [])

        all_tree_numbers.extend(tree_numbers)

    return all_tree_numbers


def get_mesh_tree_numbers(
    mesh_list: Optional[Union[str, List[str]]],
    ui_index: Dict[str, List[str]],
    name_index: Optional[Dict[str, List[str]]] = None,
    mesh_terms: Optional[Union[str, List[str]]] = None,
    sep: str = _SEP,
    deduplicate: bool = True,
) -> List[str]:
    """
    Returns the list of MeSH TreeNumbers found for a publication, using
    the SAME lookup strategy (UI -> name fallback) `classify_mesh_codes`
    uses to classify A/C/H.

    Meant for saving the TreeNumbers alongside MeSHList/MeSHTerms in the
    results CSV (the MeSHTreeNumbers column of papers_full.csv), for
    traceability: it lets one see which specific branch of the MeSH tree
    each article was assigned to, not just the final A/C/H category —
    useful, for example, to manually review why a particular article
    fell (or did not fall) into an unexpected category.

    Args:
        mesh_list: the article's MeSH UI codes (MeSHList column), in
            any of the forms _as_list accepts (a `sep`-separated
            string, a list, or None/NaN).
        ui_index: UI -> TreeNumbers index (from load_mesh_indices).
        name_index: Name -> TreeNumbers index, used as a fallback.
        mesh_terms: the corresponding readable names (MeSHTerms
            column), in the same form and the same positional order as
            mesh_list.
        sep: separator used in mesh_list/mesh_terms if they come as a
            string. Defaults to "|" (_SEP).
        deduplicate: if True (default), removes repeated TreeNumbers and
            returns them sorted alphabetically, for a more readable CSV.
            If False, keeps duplicates in the order they appear (the
            same order in which mesh_list's descriptors were walked).

    Returns:
        List of TreeNumbers (strings), e.g. ["A11.436", "B01.050..."].
        Empty list if mesh_list is empty/None, or if none of its codes
        could be located in the indices.
    """
    codes = _as_list(mesh_list, sep)
    if not codes:
        return []

    terms = _as_list(mesh_terms, sep) if mesh_terms else []
    tree_numbers = _lookup_tree_numbers(codes, ui_index, name_index, terms)

    if deduplicate:
        return sorted(set(tree_numbers))
    return tree_numbers


def classify_mesh_codes(
    mesh_list: Optional[Union[str, List[str]]],
    ui_index: Dict[str, List[str]],
    name_index: Optional[Dict[str, List[str]]] = None,
    mesh_terms: Optional[Union[str, List[str]]] = None,
    sep: str = _SEP,
) -> FrozenSet[str]:
    """
    The module's main function: classifies a SINGLE article (from its
    MeSH descriptors) into the set of translational-triangle categories
    (A/C/H) it belongs to, applying in order:

    1. Normalizes `mesh_list`/`mesh_terms` to lists of strings (with
       _as_list), accepting equally whether they arrive as
       `sep`-separated text or already as a Python list.
    2. If `mesh_list` ends up empty after normalizing (the article has
       no MeSH descriptor at all: for example, one with Status
       "NOT_FOUND" in scopus_to_pmid.py, which never had PubMed metadata
       to begin with), an empty frozenset() is returned directly,
       without attempting any lookup.
    3. Looks up the TreeNumbers of every code (with _lookup_tree_numbers,
       the UI -> name-fallback strategy described at the top of this
       file).
    4. For EACH TreeNumber found, checks which categories it belongs to
       (with _classify_tree_number) and accumulates in the `found`
       dictionary whether the article, overall, has already touched
       each category at least once (it is enough for a SINGLE
       TreeNumber, out of ALL of the article's descriptors, to fall
       under a category for that category to count as present).
    5. Returns, as a frozenset, only the categories that ended up
       True — that is, the set of A/C/H detected for this article. An
       article with MeSH descriptors that does not touch any of the
       triangle's three branches still returns an empty frozenset()
       (main.py later labels that case "UNCATEGORIZED", distinguishing
       it from "NOT_FOUND" thanks to the fact that it did have
       something in MeSHList).

    A frozenset (not a plain set or a list) is used as the return type
    because the result, in main.py, is temporarily stored as a
    DataFrame column value before being converted to text (see
    _classify_papers_full/_label_category in main.py): a frozenset is
    immutable and "hashable", which avoids problems if pandas ever
    needed to treat it as a comparable value or use it as a key.

    Args:
        mesh_list: the article's MeSH UI codes (the MeSHList column of
            papers_full.csv), in any of the forms _as_list accepts.
        ui_index: UI -> TreeNumbers index (from load_mesh_indices).
        name_index: Name -> TreeNumbers index, used as a fallback when a
            UI code does not appear in ui_index.
        mesh_terms: the corresponding readable names (MeSHTerms
            column), in the same form and the same positional order as
            mesh_list. Needed only to be able to apply the name
            fallback when required.
        sep: separator used in mesh_list/mesh_terms if they come as a
            string. Defaults to "|" (_SEP).

    Returns:
        A frozenset with the detected category letters, from among "A",
        "C", "H" (it can contain none, one, two, or all three at once).
        Empty frozenset if mesh_list is empty/None, or if none of its
        descriptors falls under any of the triangle's three branches.
    """
    codes = _as_list(mesh_list, sep)
    if not codes:
        return frozenset()

    terms = _as_list(mesh_terms, sep) if mesh_terms else []
    tree_numbers = _lookup_tree_numbers(codes, ui_index, name_index, terms)

    found = {"A": False, "C": False, "H": False}
    for tree_number in tree_numbers:
        hits = _classify_tree_number(tree_number)
        for category, matched in hits.items():
            found[category] = found[category] or matched

    return frozenset(cat for cat, matched in found.items() if matched)
