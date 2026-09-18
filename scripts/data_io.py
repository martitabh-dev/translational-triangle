"""
data_io.py

This module centralizes ALL of the pipeline's CSV reading and writing,
so that no other script has to worry about how pandas interprets the
type of each column. The reason this module exists is a very specific
pandas problem: when a DataFrame is saved to CSV and read back later
(something this pipeline does constantly, since every stage re-reads
and re-writes the same papers_full.csv), columns that should be text
identifiers (ScopusID, PMID...) or integers (Year...) can end up
converted to float (showing up as "12345678.0" instead of "12345678")
or lose their real boolean type. This module solves that problem by
ALWAYS applying the same type cleanup, both on read and on write, so
that an identifier or a year always looks the same no matter how many
times the file has been saved and reloaded.

Three families of "delicate" columns are distinguished, each with its
own cleanup function:
  - Text identifiers (ScopusID, PMID, and their Citing*/Parent*
    variants from citations_details.csv): must always stay as text,
    never as a number, because they are never used to compute anything,
    only to compare for equality.
  - Integers (Year, CitingYear, Generation): must stay as genuine
    integer numbers (not text, not float), because they ARE used in
    arithmetic (subtracting years, comparing generations).
  - Boolean (PathToH): must stay as a real Python/pandas bool, not as
    the "True"/"False" text left behind after a CSV save.

Separator convention: every file the pipeline itself generates and
reads back on its own (the output of the Scopus/PubMed resolution,
citations_details.csv, papers_full.csv, etc.) always uses ';' as the
column separator and ',' as the decimal separator — these are the
default values of read_csv_clean/to_csv_clean, so simply not specifying
anything when calling them keeps everything consistent with itself. The
only exception is the very first input CSV the user themselves provides
(e.g. a Scopus export downloaded by hand), which can come separated by
';' or by ',' depending on the regional settings it was generated with:
for THAT specific case there is `read_input_csv`, which detects on its
own, by trial, which of the two separators is the right one.
"""

from __future__ import annotations

import pandas as pd


class InputError(Exception):
    """Exception for configuration or format errors that come from the
    USER'S SIDE (they are not bugs in the code): for example, the input
    CSV missing one of the expected columns, or a request to generate
    the per-author summary over a papers_full.csv that does not yet have
    the forward-citation columns computed.

    A custom exception is used, distinct from a generic ValueError,
    precisely so it can be caught selectively at the program's entry
    point (the `if __name__ == "__main__":` block in main.py): catching
    it there prints only the error message as is, without the full call
    stack (traceback) Python shows by default. This is intentional: a
    traceback full of function names and line numbers is useful
    information for debugging a bug, but for a "you're missing this
    column in your CSV" error it only manages to scare the user and
    bury the message they actually need to read."""


# Columns that must ALWAYS be treated as text, never as a number, both
# on read and on write. This includes both the names used in
# papers_full.csv (ScopusID, PMID) and their equivalents in
# citations_details.csv (forward_citations.py / check_path.py), which
# carry the Citing/Parent prefix because in that file each row describes
# a relationship between a "parent" article (the one being cited) and a
# "citing" article, and each one needs its own identifier.
# Without this treatment, any of these columns would come back as float
# (e.g. "12345678.0" instead of "12345678") as soon as pandas
# reinterpreted it after a save/reload, because a digits-only column
# with some empty value (NaN) is interpreted as numeric by default.
_STRING_ID_COLUMNS = (
    "ScopusID", "PMID",
    "CitingScopusID", "CitingPMID",
    "ParentScopusID", "ParentPMID",
    "CitingNodeID", "ParentNodeID",
)
# Columns that must be treated as INTEGER numbers (pandas' Int64 type,
# not numpy's native int64): Year and CitingYear are publication years,
# which get subtracted from each other (e.g. CitingYear - root_year to
# compute Translational Years); Generation is the citation "hop" number
# at which an article was discovered during forward_citations.py's BFS.
# Int64 (capitalized) is used specifically because it is the only
# pandas integer type that supports null values (NaN) while still being
# a genuine integer — numpy's int64 does not allow NaN, and leaving it
# as float64 to be able to have NaN would lose the guarantee that the
# value is always an exact whole number.
_INT_COLUMNS = ("Year", "CitingYear", "Generation")
# The pipeline's only boolean column: indicates whether an article is
# part of the shortest citation path to the H article that gives it its
# TD (Translational Distance). After being saved to CSV and read back,
# pandas brings it back as the text "True"/"False" (a string), not as a
# real bool, so it needs its own cleanup in order to be used in boolean
# comparisons (if row["PathToH"]:) without a "False" text (which is
# "truthy" in Python, being a non-empty string) being incorrectly
# evaluated as true.
_BOOL_COLUMNS = ("PathToH",)


def _clean_id_series(series: pd.Series) -> pd.Series:
    """Cleans a text-identifier column (any of those listed in
    _STRING_ID_COLUMNS).

    Steps applied, in order:
      1. Forces the whole column to text (str), whatever its starting
         type (float, int, object...).
      2. Using a regular expression, strips any decimal suffix ",0" or
         ".0" left stuck at the end of the value: this happens when
         pandas interpreted the column as float at some earlier point
         in the pipeline (for example, if it had empty values) and,
         when converting it back to text, every whole number drags that
         ".0" along (or ",0" if the CSV used a decimal comma).
      3. The literal text "nan" (which appears when an originally
         empty/NaN value is forced to text in step 1) is replaced back
         with pd.NA, pandas' real "empty" value — that way the gap is
         still recognized as a missing value (for .isna(), for example)
         and not as the four-letter text "nan".

    Args:
        series: the column (pd.Series) to clean.

    Returns:
        The same column, already as clean text, with gaps as pd.NA
        instead of as the string "nan".
    """
    cleaned = series.astype(str).str.replace(r"[,.]0$", "", regex=True)
    return cleaned.replace("nan", pd.NA)


def _clean_int_series(series: pd.Series) -> pd.Series:
    """Cleans an integer numeric column (any of those listed in
    _INT_COLUMNS: Year, CitingYear, Generation).

    It follows the same first stretch as _clean_id_series (force to
    text, strip the leftover ",0"/".0" suffix, and replace the "nan"
    string with pd.NA), but instead of returning the result as text, in
    the last step it converts it to a number with pd.to_numeric (with
    errors="coerce", so that any value that cannot be interpreted as a
    number becomes NaN instead of raising an error) and finally forces
    the final type to Int64 — pandas' integer type that supports null
    values (see the comment next to _INT_COLUMNS above for why Int64
    and not int64/float64).

    The final result is a REAL integer, not text, precisely so it can
    be used directly in numeric comparisons and subtractions (e.g.
    `CitingYear - root_year`) without having to convert it first every
    time it is needed in another module.

    Args:
        series: the column (pd.Series) to clean.

    Returns:
        The same column, already as Int64 (pandas integer with null
        support).
    """
    cleaned = series.astype(str).str.replace(r"[,.]0$", "", regex=True)
    cleaned = cleaned.replace("nan", pd.NA)
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")


def _clean_bool_series(series: pd.Series) -> pd.Series:
    """Cleans the PathToH boolean column, normalizing it to a real
    Python/pandas bool, whatever its starting form.

    It is considered True only in two cases:
      - The value is already the Python bool `True` (for example, if
        this function is called on a column that has never gone through
        a CSV, freshly computed in memory).
      - The value, converted to text, lowercased and with extra
        whitespace stripped, is "true" or "1" (to recognize both
        "True"/"TRUE" and a possible "1" if the column arrived as a
        numeric 0/1).

    ANY other case (the Python bool `False`, the text "false" in any
    combination of upper/lower case, an empty cell or NaN, or any other
    unexpected text) is treated as False. This matters because, without
    this function, a cell holding the text "False" read from a CSV
    would evaluate as true in a Python `if` (since "False" is a
    non-empty string, and every non-empty string is "truthy").

    Args:
        series: the PathToH column (pd.Series) to clean.

    Returns:
        The same column, already as a real Python bool (True/False,
        with no null values: a gap/NaN becomes False).
    """
    return series.apply(lambda v: v is True or str(v).strip().lower() in ("true", "1"))


def read_csv_clean(path, sep: str = ";", decimal: str = ",", **kwargs) -> pd.DataFrame:
    """Reads a CSV with pandas (pd.read_csv) and immediately applies to
    it the type cleanup for the three families of delicate columns
    (text identifiers, integers, boolean) described at the top of this
    file.

    The default values of `sep`/`decimal` (';' and ',') are the ones
    the pipeline itself uses internally for EVERY file it generates and
    reads back on its own; for the user's original input CSV, which may
    come with a different separator, `read_input_csv` is used instead
    (see below), not this function directly.

    Identifier columns are forced to text (dtype=str) already from the
    call to pd.read_csv itself, even before applying _clean_id_series:
    this prevents pandas, while parsing the CSV, from interpreting a
    digits-only column as numeric from the start (which would already
    cause, for example, the loss of leading zeros if there were any).

    Args:
        path: path of the CSV to read (any type pd.read_csv accepts:
            str, Path, or a file-like object).
        sep: the CSV's column separator. Defaults to ';'.
        decimal: the decimal separator used in the CSV's numbers.
            Defaults to ','.
        **kwargs: any other extra argument is forwarded as is to
            pd.read_csv (for example, to be able to pass an extra
            `dtype` or `usecols` if some caller needed it).

    Returns:
        The already-read DataFrame with the delicate columns (whichever
        ones are present in the CSV; not all of them are required to
        appear) already cleaned.
    """
    # The `dtype` dict is built BEFORE reading, to force each identifier
    # column to text (str) from the read itself. If the caller also
    # passed their own `dtype` in kwargs, it is respected and added on
    # top (so a caller could, in theory, override this default behavior
    # for some column if they needed to).
    dtype = {col: str for col in _STRING_ID_COLUMNS}
    dtype.update(kwargs.pop("dtype", {}))
    df = pd.read_csv(path, sep=sep, decimal=decimal, dtype=dtype, **kwargs)
    # Each block below walks its list of "delicate" columns and only
    # acts on the ones that are actually present in this particular CSV
    # (`if col in df.columns`): not every CSV in the pipeline has every
    # column (for example, a papers_full.csv freshly created by
    # scopus_to_pmid.py does not yet have PathToH, which only appears
    # after forward_citations.py has run).
    for col in _STRING_ID_COLUMNS:
        if col in df.columns:
            df[col] = _clean_id_series(df[col])
    for col in _INT_COLUMNS:
        if col in df.columns:
            df[col] = _clean_int_series(df[col])
    for col in _BOOL_COLUMNS:
        if col in df.columns:
            df[col] = _clean_bool_series(df[col])
    return df


def read_input_csv(path, **kwargs) -> pd.DataFrame:
    """Reads the ORIGINAL input CSV the user provides (for example, a
    Scopus export downloaded from their website), whose column
    separator cannot be assumed ahead of time: it may come separated by
    ';' (typical of Excel configured for Spanish, with ',' as the
    decimal separator) or by ',' (the "international" standard CSV
    format, with '.' as the decimal separator), depending on the
    regional settings the file was generated with.

    Strategy to detect which of the two it is: it first tries reading
    with ';'/',' (read_csv_clean's default). If the real separator was
    actually ',' instead of ';', pandas does not raise any error when
    reading with ';' — it simply interprets the WHOLE row as a single
    column, because it never finds the ';' character to split the
    values on. That is why the signal that the tried separator was NOT
    the right one is that the resulting DataFrame has only one column
    (`df.shape[1] > 1` is false): in that case, that attempt is
    discarded and it reads again, this time with ',' as the separator
    and '.' as the decimal. If instead the first read raised some
    exception (a rarer case, but possible with malformed files), that
    is also ignored and it retries with ',' all the same.

    Deliberately, this function does NOT check the name of any specific
    column: the user may have named their ScopusID, researcher and
    publication-year columns however they like (see main.py's
    --scopus-id-column/--author-id-column/--year-column arguments,
    which are what later locate and rename them once the DataFrame is
    already in memory).

    This function is meant SOLELY for this first input CSV. Everything
    the pipeline itself writes and reads back from that point on
    (papers_full.csv, citations_details.csv, etc.) is always already
    internally unified to ';'/',' from the moment it is first saved, so
    for those files it is enough to use read_csv_clean/to_csv_clean
    directly, with no need to detect anything.

    Args:
        path: path of the original input CSV.
        **kwargs: forwarded as is to read_csv_clean on either of the two
            attempts.

    Returns:
        The DataFrame already read with the correct separator and with
        the same type cleanup read_csv_clean applies.
    """
    try:
        df = read_csv_clean(path, sep=";", decimal=",", **kwargs)
        if df.shape[1] > 1:
            return df
    except Exception:
        # Any failure during the first attempt (e.g. a parsing error if
        # the file were actually malformed for ';') is simply ignored:
        # it just moves on to retrying with ',' further below, instead
        # of propagating the exception upward.
        pass
    return read_csv_clean(path, sep=",", decimal=".", **kwargs)


def to_csv_clean(df: pd.DataFrame, path, sep: str = ";", decimal: str = ",", **kwargs) -> None:
    """Saves a DataFrame to CSV (pd.to_csv), reapplying right before
    that, on a COPY of the received DataFrame, the same type cleanup as
    read_csv_clean (text identifiers, integers, boolean).

    This re-cleanup right before saving is necessary because, even if
    the DataFrame already came clean from an earlier read, it is common
    for operations to be applied to it at some intermediate point in the
    pipeline (an arithmetic operation, a `merge` with another DataFrame,
    a `groupby`...) that can turn, for example, an identifier column
    back into float, or an integer column into float from the mere
    presence of some NaN. Cleaning here, right before writing,
    guarantees that the CSV left on disk always has the correct type,
    without depending on every intermediate function in the pipeline
    remembering to maintain it on its own.

    It works on `df.copy()` (never on the original DataFrame the caller
    passed in) so as not to unexpectedly modify, as a side effect of
    saving to disk, a DataFrame the calling code might keep using
    afterward.

    Args:
        df: the DataFrame to save.
        path: destination path of the CSV.
        sep: column separator to use in the output file. Defaults to
            ';'.
        decimal: decimal separator to use in the output file. Defaults
            to ','.
        **kwargs: any other extra argument is forwarded as is to
            pd.to_csv (for example, `mode="a"` if some caller needed to
            append rows to an existing file instead of overwriting it).
    """
    df = df.copy()
    for col in _STRING_ID_COLUMNS:
        if col in df.columns:
            df[col] = _clean_id_series(df[col])
    for col in _INT_COLUMNS:
        if col in df.columns:
            df[col] = _clean_int_series(df[col])
    for col in _BOOL_COLUMNS:
        if col in df.columns:
            df[col] = _clean_bool_series(df[col])
    # index=False: by default, pandas would also write the DataFrame's
    # internal numeric index (0, 1, 2...) as if it were another column
    # of the CSV; it is deliberately dropped here because that index
    # has no meaning for the rest of the pipeline (each row's real
    # identifiers are ScopusID/PMID, not the DataFrame's in-memory
    # position).
    df.to_csv(path, sep=sep, decimal=decimal, index=False, **kwargs)
