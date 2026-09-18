"""
visualization.py

Generates authors_triangle.html: a self-contained HTML page (no external
dependencies besides the Plotly.js library, loaded from a CDN) that
draws the A-C-H translational triangle and, inside it, one point per
researcher (AuthorID) at the mean position of their articles (xMean,
yMean, computed by metrics.compute_metrics_by_author).

The HTML is built in two layers:
  1. A static HTML/CSS/JavaScript template
     (_RESEARCHERS_TRIANGLE_TEMPLATE, using string.Template with
     "$name" placeholders), which defines the page, its controls and
     all the interaction logic with Plotly.
  2. The concrete data of a given run (one JSON record per researcher,
     plus the vertex coordinates, the gradient colors, etc.), which is
     inserted into that template with Template.substitute (see
     plot_authors_triangle).

The resulting page offers two viewing modes, switchable with the
buttons at the top:
  - "Top N mode": a slider showing the N researchers with the most
    articles (NArticles), from highest to lowest.
  - "Manual selection mode": a search box to add specific researchers
    by their AuthorID or their AuthorNumber, shown labeled by their ID
    directly on the point.

Each point is colored according to its TCMean (the researcher's mean
Translational Closeness) with a purple (worse, low TC) -> green
(better, high TC) gradient, and drawn with a size proportional to the
square root of its NArticles (so that the point's AREA, not its
diameter, is proportional to the number of articles). Researchers with
no article that has a path to H (TCMean at 0 or NaN) are painted a
neutral gray, outside the gradient, since it makes no sense to assign
them a position within a scale built over positive TC values.

Researchers with no valid (xMean, yMean) at all are not drawn in the
triangle (there is no position to draw them at), but their count is
still shown in a banner right below the page title (n_no_position),
so a reader doesn't mistake a missing researcher for a bug when their
own count of researchers doesn't match the number of points on the
page.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
from .geometry import VERTICES
from matplotlib.colors import LinearSegmentedColormap
import json
from string import Template


def _tc_color_position(value: float, gamma: float) -> float:
    """
    Converts a TC (Translational Closeness) value, already expected in
    the [0, 1] range, into a position within the color gradient (also
    in [0, 1]), applying a gamma correction.

    Why a gamma correction is needed: TC is defined as 1/TD (see
    metrics.py), so its possible values are not spread uniformly over
    [0, 1] — they cluster heavily toward the low end (TD=1 -> TC=1.0,
    TD=2 -> TC=0.5, TD=3 -> TC=0.33, TD=4 -> TC=0.25...). Without
    correcting for this, almost every point would fall in the same
    color band of the gradient, and the visual difference between
    "close to H" and "very close to H" would be hard to tell apart.
    Raising the value to `gamma` (with gamma > 1, like
    plot_authors_triangle's default of 1.5) visually spreads out those
    low values, giving them more of the gradient's range.

    Args:
        value: the researcher's TC value, clamped to [0, 1] before
            applying the power (as a safety measure, although in
            practice TC is already in that range by construction).
        gamma: the correction exponent. gamma=1 leaves the value
            unchanged; gamma>1 compresses low values toward the lower
            end of the gradient (spreading them apart more visually).

    Returns:
        Position within the gradient, in [0, 1], ready to be passed
        directly to the colormap (cmap(pos)) or to the vertical
        position of the gradient bar's tick marks in the HTML (see
        tick_html in plot_authors_triangle).
    """
    value = max(0.0, min(1.0, value))
    return value ** gamma

# HTML/CSS/JavaScript template for the whole page. Uses string.Template
# ("$name" placeholders, not an f-string or .format) precisely because
# the template is full of the curly braces { } that CSS and JavaScript
# use, which with an f-string would have to be escaped twice ({{ }})
# everywhere; with Template the only thing to avoid is a loose "$"
# symbol in the HTML/JS (none appears here), and the values are
# inserted further down with .substitute(...) in plot_authors_triangle.
# The embedded JavaScript assumes Plotly.js is already loaded (see the
# <script src=...> below) and uses Plotly.react to redraw the chart
# every time the mode, the "Top N" slider, or the manual researcher
# selection changes.
_RESEARCHERS_TRIANGLE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<Title>$Title</Title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; }
  #layout { display: flex; gap: 24px; align-items: flex-start; }
  #controls { margin-bottom: 12px; }
  #modeButtons { margin-bottom: 14px; }
  #modeButtons button {
    padding: 6px 14px; font-size: 14px; cursor: pointer;
    border: 1px solid #888; background: #eee; border-radius: 4px 0 0 4px;
  }
  #modeButtons button:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  #modeButtons button.active { background: #4a90d9; color: white; border-color: #2a6cb0; }

  #topNRow, #searchRow { display: none; }
  #topNRow.visible, #searchRow.visible { display: block; }

  #controls label { font-size: 14px; }
  #chart { width: 900px; height: 900px; }
  #searchResult { font-size: 13px; color: #555; margin-top: 4px; min-height: 18px; }
  input[type="text"] { padding: 4px 8px; font-size: 14px; width: 220px; }
  input[type="range"] { width: 220px; }
  #searchBox { display: flex; gap: 6px; align-items: center; }
  #searchBox button, #clearBtn { padding: 4px 10px; font-size: 14px; cursor: pointer; }
  #selectedList { margin-top: 8px; max-width: 500px; }
  .chip {
    display: inline-block; background: #eee; border: 1px solid #bbb; border-radius: 12px;
    padding: 2px 6px 2px 10px; margin: 2px; font-size: 12px;
  }
  .chip button {
    border: none; background: none; cursor: pointer; font-size: 13px; margin-left: 4px; color: #900;
  }
  #clearRow { margin-top: 8px; }

  #colorbarPanel { width: 90px; margin-top: 60px; }
  #colorbarTitle { font-size: 12px; font-weight: bold; text-align: center; margin-bottom: 8px; }
  #noTcInfo { font-size: 12px; color: #333; text-align: center; margin-top: 14px; max-width: 90px; }
  #grayBox {
    width: 14px; height: 14px; background: #b0b0b0; border: 1px solid #888;
    display: inline-block; vertical-align: middle; margin-right: 4px;
  }
  #noPositionBanner {
    font-size: 14px; color: #333; background: #eef5fc; border-left: 4px solid #4a90d9;
    padding: 10px 14px; margin-bottom: 16px; max-width: 700px;
  }
</style>
</head>
<body>
  <h2>$Title</h2>
  $no_position_banner
  <div id="controls">
    <div id="modeButtons">
      <button id="modeTopNBtn" class="active">Top N mode (slider)</button>
      <button id="modeSearchBtn">Manual selection mode</button>
    </div>

    <div id="topNRow" class="visible">
      <label for="topN">Researchers shown (by number of articles): <b id="topNValue">$default_top_n</b> / $max_n</label><br>
      <input type="range" id="topN" min="1" max="$max_n" value="$default_top_n">
    </div>

    <div id="searchRow">
      <label for="search">Add researcher (AuthorID or AuthorNumber):</label>
      <div id="searchBox">
        <input type="text" id="search" placeholder="e.g. R_1QoTo7UY8J o 118">
        <button id="addBtn">Add</button>
      </div>
      <div id="searchResult"></div>
      <div id="selectedList"></div>
      <div id="clearRow">
        <button id="clearBtn">Deselect all</button>
      </div>
    </div>
  </div>
  <div id="layout">
    <div id="chart"></div>
    <div id="colorbarPanel">
      <div id="colorbarTitle">Mean TC<br>(Translational<br>Closeness)</div>
      <div style="position: relative; height: 320px; width: 24px; margin: 0 auto;">
        <div style="width: 24px; height: 320px; background: linear-gradient(to top, $gradient_stops); border: 1px solid #888;"></div>
        $tick_html
      </div>
      <div id="noTcInfo">
        <span id="grayBox"></span>TC=0: n=$n_zero_or_nan
      </div>
    </div>
  </div>

<script>
const DATA = $data_json;
const VERTICES = { A: [$ax, $ay], C: [$cx, $cy], H: [$hx, $hy] };
let selected = [];
let mode = "topN"; // "topN" | "search"

function triangleShapeTrace() {
  return {
    x: [VERTICES.A[0], VERTICES.C[0], VERTICES.H[0], VERTICES.A[0]],
    y: [VERTICES.A[1], VERTICES.C[1], VERTICES.H[1], VERTICES.A[1]],
    mode: "lines", line: {color: "black", width: 1.5},
    hoverinfo: "skip", showlegend: false,
  };
}

function hoverText(d) {
  return "AuthorID: " + d.name +
    "<br>AuthorNumber: " + d.id +
    "<br>NArticles: " + d.n_articles +
    "<br>TF: " + (d.tf === null ? "-" : d.tf) +
    "<br>TDMean: " + (d.td === null ? "-" : d.td) +
    "<br>TYMean: " + (d.ty === null ? "-" : d.ty) +
    "<br>TCMean: " + (d.tc === null ? "-" : d.tc);
}

const MIN_PX = $min_marker_px, MAX_PX = $max_marker_px;
function sizeFor(d, maxN) {
  const frac = maxN > 0 ? Math.sqrt(d.n_articles / maxN) : 0;
  return MIN_PX + frac * (MAX_PX - MIN_PX);
}

function setMode(newMode) {
  mode = newMode;
  document.getElementById("modeTopNBtn").classList.toggle("active", mode === "topN");
  document.getElementById("modeSearchBtn").classList.toggle("active", mode === "search");
  document.getElementById("topNRow").classList.toggle("visible", mode === "topN");
  document.getElementById("searchRow").classList.toggle("visible", mode === "search");
  render();
}

function findMatches(query) {
  query = query.trim().toLowerCase();
  if (!query) return [];
  return DATA.filter(d =>
    String(d.id).toLowerCase() === query ||
    (d.name && d.name.toLowerCase().includes(query))
  );
}

function addSelected() {
  const query = document.getElementById("search").value;
  const matches = findMatches(query);
  const resultEl = document.getElementById("searchResult");
  if (matches.length === 0) {
    resultEl.textContent = 'No researcher found for "' + query + '".';
    return;
  }
  matches.forEach(m => {
    if (!selected.some(s => s.id === m.id)) selected.push(m);
  });
  document.getElementById("search").value = "";
  resultEl.textContent = "";
  renderSelectedList();
  render();
}

function removeSelected(id) {
  selected = selected.filter(d => d.id !== id);
  renderSelectedList();
  render();
}

function clearSelected() {
  selected = [];
  renderSelectedList();
  render();
}

function renderSelectedList() {
  const container = document.getElementById("selectedList");
  container.innerHTML = "";
  selected.forEach(d => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = d.name + " (ID " + d.id + ")";
    const btn = document.createElement("button");
    btn.textContent = "×";
    btn.onclick = () => removeSelected(d.id);
    chip.appendChild(btn);
    container.appendChild(chip);
  });
}

function render() {
  const sorted = [...DATA].sort((a, b) => b.n_articles - a.n_articles);
  const maxN = sorted.length ? sorted[0].n_articles : 1;

  let shown;
  if (mode === "search") {
    shown = [...selected];
  } else {
    const topN = parseInt(document.getElementById("topN").value, 10);
    document.getElementById("topNValue").textContent = topN;
    shown = sorted.slice(0, topN);
  }

  const trace = {
    x: shown.map(d => d.x), y: shown.map(d => d.y),
    mode: "markers+text",
    text: mode === "search" ? shown.map(d => String(d.id)) : [],
    textposition: "middle center",
    textfont: {size: 11, color: "black"},
    marker: {
      size: shown.map(d => sizeFor(d, maxN)),
      color: shown.map(d => d.color),
      line: {width: 0.5, color: "rgba(0,0,0,0.3)"},
      opacity: 0.85,
    },
    hovertext: shown.map(hoverText), hoverinfo: "text", showlegend: false,
  };

  const layout = {
    xaxis: {visible: false, scaleanchor: "y", scaleratio: 1},
    yaxis: {visible: false},
    plot_bgcolor: "white", width: 900, height: 900,
    margin: {t: 40, b: 40, l: 40, r: 40},
    annotations: [
      {x: VERTICES.A[0], y: VERTICES.A[1], text: "<b>A</b>", showarrow: false, font: {size: 18}, xshift: -14},
      {x: VERTICES.C[0], y: VERTICES.C[1], text: "<b>C</b>", showarrow: false, font: {size: 18}, yshift: 14},
      {x: VERTICES.H[0], y: VERTICES.H[1], text: "<b>H</b>", showarrow: false, font: {size: 18}, xshift: 14},
    ],
  };

  Plotly.react("chart", [triangleShapeTrace(), trace], layout, {displayModeBar: true});
}

document.getElementById("modeTopNBtn").addEventListener("click", () => setMode("topN"));
document.getElementById("modeSearchBtn").addEventListener("click", () => setMode("search"));
document.getElementById("topN").addEventListener("input", render);
document.getElementById("addBtn").addEventListener("click", addSelected);
document.getElementById("search").addEventListener("keydown", (e) => { if (e.key === "Enter") addSelected(); });
document.getElementById("clearBtn").addEventListener("click", clearSelected);
render();
</script>
</body>
</html>
""")

# Same colors as the reference legend: high TD (worse, 4) in purple,
# low TD (better, 1) in green, passing through magenta-red-orange-yellow.
_TC_COLORS = [
    "#662D91",  # purple -> low TC (worse, never/far from H)
    "#EC008C",  # magenta
    "#ED1C24",  # red
    "#F7941D",  # orange
    "#FFF200",  # yellow
    "#39B54A",  # green  -> high TC (better, close to H)
]
_TC_CMAP = LinearSegmentedColormap.from_list("tc_gradient", _TC_COLORS)
# Color for researchers with no article reaching H (TDMean = NaN): they
# fall outside the gradient, since averaging over NaN is not possible.
_NO_H_COLOR = "#B0B0B0"

def plot_authors_triangle(
    authors_df: pd.DataFrame,
    x_col: str = "xMean",
    y_col: str = "yMean",
    size_col: str = "NArticles",
    color_col: str = "TCMean",
    id_col: str = "AuthorNumber",
    name_col: str = "AuthorID",
    cmap_name=_TC_CMAP,
    Title: str = "Translational triangle by researcher",
    save_path: Optional[str] = None,
    default_top_n: int = 20,
    min_marker_px: float = 5.0,
    max_marker_px: float = 90.0,
    tc_gamma: float = 1.5,
) -> str:
    """
    Generates the interactive HTML page of the per-researcher
    translational triangle, and optionally saves it to disk.

    Takes `authors_df` (the output of metrics.compute_metrics_by_author,
    one row per AuthorID) and, for each researcher with a valid
    position, builds a JSON record with everything the page needs to
    draw it and for its mouse-hover tooltip: position (x, y), size
    (derived from NArticles), color (derived from TCMean), and the
    TF/TDMean/TYMean/TCMean metrics already rounded down to native
    Python types (float/None) so json.dumps can serialize them. Those
    records, together with the triangle's vertex coordinates and the
    visual parameters (size range, gradient colors, etc.), are inserted
    into _RESEARCHERS_TRIANGLE_TEMPLATE to produce the final HTML.

    Researchers excluded from `plotted` (and therefore invisible in the
    triangle, although if their TCMean is 0/NaN they are still counted
    in n_zero_or_nan, which is shown in the legend panel): those without
    valid xMean/yMean (e.g. if compute_metrics_by_author could not
    compute a position for any of their articles). Their count
    (n_no_position) is shown as a banner right below the page title, so
    a reader whose own researcher count doesn't match the number of
    points on the page has an explanation for the gap immediately,
    instead of assuming something is missing or broken. The banner is
    only rendered when n_no_position > 0.

    Args:
        authors_df: per-researcher DataFrame, typically the output of
            compute_metrics_by_author.
        x_col, y_col: columns with the researcher's mean position in
            the triangle.
        size_col: column used for the point's size (the researcher's
            number of articles).
        color_col: column used for the point's color (mean TC).
        id_col: column with the short numeric identifier shown on the
            point in "Manual selection" mode, and also accepted as a
            search term.
        name_col: column with the researcher's real identifier
            (AuthorID), shown in the tooltip and in the selected-items
            list.
        cmap_name: matplotlib colormap to use for the color gradient, as
            a registered name (str) or as an already-built Colormap
            object; defaults to _TC_CMAP (the purple -> green gradient
            defined in this module).
        Title: the page's title, used both in the HTML <title> and in
            the visible <h2> heading.
        save_path: if given, the path to write the generated HTML to
            (utf-8 encoded). If None, the HTML is only returned, with
            nothing written to disk.
        default_top_n: number of researchers shown initially in "Top N"
            mode when the page loads (clamped to the total number of
            available researchers if there are fewer).
        min_marker_px, max_marker_px: diameter range, in pixels, of the
            drawn points: the researcher with the fewest articles is
            drawn with diameter min_marker_px, and the one with the most
            articles (within the set shown at any given moment) with
            diameter max_marker_px; the rest are interpolated according
            to the square root of their number of articles (see sizeFor
            in the template's JavaScript), so that it is the point's
            AREA, not its diameter, that ends up proportional to
            NArticles.
        tc_gamma: gamma correction exponent applied when positioning
            each TC within the color gradient (see _tc_color_position)
            and when computing the vertical position of the gradient
            bar's numeric tick marks (tick_html).

    Returns:
        The page's full HTML, as a string. If `save_path` is not None,
        that same content is also written to that path as a side
        effect.
    """
    plotted = authors_df.dropna(subset=[x_col, y_col]).copy()

    if isinstance(cmap_name, mcolors.Colormap):
        cmap = cmap_name
    else:
        cmap = matplotlib.colormaps[cmap_name]

    def _to_hex(value):
        if pd.isna(value) or not np.isfinite(value) or value == 0:
            return "#b0b0b0"
        pos = _tc_color_position(value, tc_gamma)
        return mcolors.to_hex(cmap(pos))

    def _is_zero_or_nan_tc(value):
        return pd.isna(value) or not np.isfinite(value) or value == 0

    # Counted over the WHOLE authors_df (not only `plotted`), to also
    # include researchers without xMean/yMean, who never even get drawn
    # in the triangle but can equally have TC=0/NaN.
    n_zero_or_nan = 0
    n_zero_or_nan = int(authors_df[color_col].apply(_is_zero_or_nan_tc).sum())
    n_no_position = int(authors_df[[x_col, y_col]].isna().any(axis=1).sum())

    # Shown as a banner right under the page title (not buried in the
    # legend panel), so a reader whose own researcher count doesn't
    # match the number of points on the page finds the explanation
    # immediately instead of assuming something is missing or broken.
    # Only rendered when there actually are excluded researchers.
    if n_no_position > 0:
        no_position_banner = (
            '<div id="noPositionBanner">'
            f"{n_no_position} researcher(s) out of {len(authors_df)} are not shown in the triangle: "
            "they have no valid (x, y) position, because all of their articles are "
            "UNCATEGORIZED or NOT_FOUND (see summary_report.txt for the list)."
            "</div>"
        )
    else:
        no_position_banner = ""

    records = []
    for _, row in plotted.iterrows():
        tc_value = row.get(color_col)
        records.append({
            "id": int(row[id_col]) if pd.notna(row[id_col]) else None,
            "name": str(row.get(name_col, "")),
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "n_articles": float(row.get(size_col, 0) or 0),
            "tc": None if pd.isna(tc_value) else float(tc_value),
            "td": None if pd.isna(row.get("TDMean")) else float(row["TDMean"]),
            "ty": None if pd.isna(row.get("TYMean")) else float(row["TYMean"]),
            "tf": None if pd.isna(row.get("TF")) else float(row["TF"]),
            "color": _to_hex(tc_value),
        })

    # CSS gradient bar with the SAME colors/stops (_TC_COLORS), evenly
    # spaced the same way LinearSegmentedColormap does by default, so
    # the bar matches the points' actual colors exactly.
    n_colors = len(_TC_COLORS)
    stops = [
        f"{color} {round(100 * i / (n_colors - 1))}%"
        for i, color in enumerate(_TC_COLORS)
    ]
    gradient_stops = ", ".join(stops)

    ax_, ay_ = VERTICES["A"]
    cx_, cy_ = VERTICES["C"]
    hx_, hy_ = VERTICES["H"]

    tick_html = "\n".join(
        f'<div class="colorbarTick" style="position:absolute; '
        f'top:{(1 - _tc_color_position(v, tc_gamma)) * 100:.1f}%; '
        f'left:30px; transform: translateY(-50%); font-size:11px;">{v}</div>'
        for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    )

    html = _RESEARCHERS_TRIANGLE_TEMPLATE.substitute(
        Title=Title,
        data_json=json.dumps(records),
        default_top_n=min(default_top_n, len(records)) if records else 0,
        max_n=max(len(records), 1),
        ax=ax_, ay=ay_, cx=cx_, cy=cy_, hx=hx_, hy=hy_,
        gradient_stops=gradient_stops,
        n_zero_or_nan=n_zero_or_nan,
        no_position_banner=no_position_banner,
        min_marker_px=min_marker_px,
        max_marker_px=max_marker_px,
        tick_html=tick_html,
    )


    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html