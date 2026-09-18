"""
geometry.py

This is the second half of pipeline STAGE 2 (the first half is
mesh_classification.py): once each article already has its
translational category (A, C, H, or any combination of them, computed
by classify_mesh_codes), this module translates that category into a
geometric (x, y) position inside an equilateral triangle, and from that
position computes a single per-article summary number: the
Translational Index (TI).

The triangle: the three vertices A (Animal), C (Cell) and H (Human)
form an equilateral triangle centered at the origin (0, 0) — that is,
the barycenter of the three vertices together falls exactly there. C
sits at the very top and A/H are placed symmetrically at the bottom
left/right (see VERTICES, below, for the exact coordinates). This
specific layout is not arbitrary: being centered at the origin is what
lets the Translational Index be computed further down as a simple
scalar projection, with no offset to subtract (see
compute_translational_index).

Design decision: the three vertices A, C and H are defined once, in
VERTICES, and ALL other possible positions (the combinations of two or
three categories at once: AC, AH, CH, ACH) are derived automatically as
the barycenter (the simple arithmetic mean of their coordinates) of the
vertices that make up that combination. There are no "hardcoded"
coordinates for each of the 8 possible categories (the 3 individual
ones + the 3 two-way combinations + ACH + the empty one): only the 3
base vertices exist, and everything else is computed on the fly with
the `barycenter` function. This has a practical advantage: if the base
vertex coordinates were ever to change, no other part of the code would
need to be touched.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# A point of the triangle, as an (x, y) tuple of floats.
Point = Tuple[float, float]

# Vertices of the Weber framework's equilateral triangle, centered at
# the origin (0, 0). Exact coordinates (using the height of an
# equilateral triangle of side 2, with C at the very top and A/H at the
# bottom, symmetric about the vertical axis):
#   - A (Animal): (-√3/2, -0.5) — bottom-left vertex.
#   - C (Cell):   ( 0.0,   1.0) — top vertex.
#   - H (Human):  ( √3/2, -0.5) — bottom-right vertex.
# The mean of the three vertices is exactly (0, 0): (-√3/2+0+√3/2)/3 = 0
# on x, and (-0.5+1-0.5)/3 = 0 on y. This centering at the origin is
# what lets compute_translational_index, further down, compute the
# projection directly as x*u0 + y*u1, with no reference point to
# subtract beforehand.
VERTICES: Dict[str, Point] = {
    "A": (-np.sqrt(3) / 2, -0.5),
    "C": (0.0, 1.0),
    "H": (np.sqrt(3) / 2, -0.5),
}


def barycenter(category: str) -> Point:
    """
    Computes the barycenter (the arithmetic mean of the x and y
    coordinates separately) of the vertices corresponding to a category
    label, which can be a single letter ("A", "C", "H"), a combination
    of two or three ("AC", "AH", "CH", "ACH"), or the special case
    "None"/empty (an article with no real category: UNCATEGORIZED or
    NOT_FOUND in main.py).

    The order of the letters within `category` does not matter for the
    result (the barycenter of "AC" and of "CA" is the same, since it is
    a mean), although main.py always generates those labels already
    ordered as A-C-H (see _category_to_str in main.py).

    Args:
        category: category label, a subset of {"A", "C", "H"} expressed
            as text (e.g. "A", "AH", "ACH"), or "None"/an empty string
            if the article does not belong to any of them.

    Returns:
        An (x, y) tuple with the barycenter of the indicated vertices.
        If `category` is "None", empty, or contains NONE of the
        recognized letters (neither "A", "C" nor "H" — for example, if
        the special label "UNCATEGORIZED" or "NOT_FOUND" arrived as
        is), (nan, nan) is returned: the article has no defined position
        in the triangle, since it does not belong to any vertex.
    """
    if not category or category == "None":
        return (float("nan"), float("nan"))

    # Keeps only the characters of `category` that are recognized
    # vertex letters (A, C, H); this makes the function tolerant of any
    # label that is not a pure combination of those three letters (such
    # as "UNCATEGORIZED" or "NOT_FOUND"), since none of its characters
    # will match a VERTICES key and `letters` will end up empty,
    # returning (nan, nan) all the same via the `if not letters` check
    # below.
    letters = [c for c in category if c in VERTICES]
    if not letters:
        return (float("nan"), float("nan"))

    xs = [VERTICES[letter][0] for letter in letters]
    ys = [VERTICES[letter][1] for letter in letters]
    return (float(np.mean(xs)), float(np.mean(ys)))


def compute_positions(categories: pd.Series) -> pd.DataFrame:
    """
    Computes (x, y) coordinates for a WHOLE column of categories at
    once (applying `barycenter` row by row), meant to be used directly
    on the Category column of a papers_full.csv DataFrame.

    Args:
        categories: pandas Series with each article's category label
            (in the form `barycenter` accepts: "A", "AH", "ACH",
            "None"...).

    Returns:
        DataFrame with two columns, "x" and "y", indexed exactly like
        `categories` (same position for the same row), so it can be
        assigned directly back onto a larger DataFrame (see
        _classify_papers_full in main.py:
        `df["x"] = positions["x"]; df["y"] = positions["y"]`).
    """
    positions = categories.apply(barycenter)
    xs = positions.apply(lambda p: p[0])
    ys = positions.apply(lambda p: p[1])
    return pd.DataFrame({"x": xs, "y": ys}, index=categories.index)


def translational_axis() -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the triangle's translational axis: the direction that goes
    from "purely preclinical" (the midpoint between A and C, i.e. the
    barycenter of the "AC" combination) to "purely human" (vertex H).
    This is the direction along which each article's Translational
    Index is measured (see compute_translational_index): the further an
    article moves away from AC and toward H along THIS specific axis,
    the more translational it is considered.

    Returns:
        Tuple (origin, unit_vector):
          - origin: the AC midpoint (barycenter of A and C), as a
            2-element numpy array [x, y]. It is computed and returned
            for completeness/traceability of the geometric calculation,
            but NOTE: compute_translational_index does NOT actually use
            it as a reference point to subtract (see the note in its
            own docstring, further below, on why that is not needed).
          - unit_vector: the vector from `origin` to H, normalized to
            unit length (divided by its own norm), so that projecting
            any point onto it can be interpreted directly as a distance
            along the axis.
    """
    ac_midpoint = np.array(barycenter("AC"))
    h_vertex = np.array(VERTICES["H"])
    axis_vector = h_vertex - ac_midpoint
    unit_vector = axis_vector / np.linalg.norm(axis_vector)
    return ac_midpoint, unit_vector

def compute_translational_index(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Computes the Translational Index (TI) of each article: the scalar
    projection of its (x, y) position — already computed by
    compute_positions — onto the AC->H translational axis (obtained from
    translational_axis), measured from the triangle's actual coordinate
    origin, the point (0, 0), which also happens to coincide exactly
    with the ACH barycenter (the mean of the three vertices A, C and H
    at once — see the comment next to VERTICES, above, on why VERTICES
    is deliberately built that way).

    Important note on the calculation: although translational_axis()
    also returns `ac_midpoint` as the axis's conceptual "origin", that
    value is discarded here (captured with `_` and not used). The
    projection is computed directly as the dot product
    `(x, y) · unit_vector`, i.e. measured from (0, 0) — NOT from
    ac_midpoint. This is correct because (0, 0) is the triangle's actual
    center (the ACH barycenter), which is the reference point that
    really matters for this framework: an article sitting exactly at
    the center of the triangle (category "ACH", touching all three
    branches at once) thus gets a TI of 0 by construction, leaning
    neither toward the preclinical extreme nor the human one.

    Args:
        x: pandas Series with each article's x coordinate (the "x"
            column returned by compute_positions).
        y: pandas Series with each article's y coordinate (the "y"
            column returned by compute_positions).

    Returns:
        pandas Series with each article's TI (same index as `x`/`y`).
        An article with no defined position (x/y at nan, because
        barycenter could not place it) also gets a TI of nan, since any
        arithmetic operation with nan propagates as nan.
    """
    _, unit_vector = translational_axis()
    return x * unit_vector[0] + y * unit_vector[1]
