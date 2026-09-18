"""
scopus_to_pmid.py

This is STAGE 1 of the pipeline (the first one that runs, and the one that
creates papers_full.csv from scratch): starting from a list of articles
identified by their ScopusID (with their AuthorID and publication year),
it resolves each one to its PubMed identifier (PMID) if it exists, and
enriches it with metadata useful for the following stages: DOI, title,
and the article's MeSH descriptors (MeSHList/MeSHTerms), which is what
mesh_classification.py will use to decide its A/C/H category.

PMID resolution cascade (for each article, in this order, until finding
one that works):
  1. The Scopus API (Abstract Retrieval) is queried first with the
     ScopusID: if Scopus ALREADY knows the PMID for that article, it is
     used directly (Status "SCOPUS_PMID") and nothing further needs to
     be searched in PubMed to identify it.
  2. If Scopus does not provide the PMID but does provide a DOI, that
     DOI is searched in PubMed (esearch) and it is verified that the
     candidate found is really the same article (Status "DOI_MATCH").
  3. If there is no DOI either (or the DOI search finds nothing), the
     same process is repeated but searching by title (Status
     "TITLE_MATCH").
  4. If none of the three routes works, the article is left with
     Status "NOT_FOUND": it could not be identified in PubMed at all
     (this does not prevent the rest of the pipeline from processing
     it; it will simply have no PMID or MeSH metadata).

Once the PMIDs of all articles have been identified, a second pass
downloads in bulk (via efetch, in batches) their complete PubMed
metadata — above all the MeSH descriptors, which is the data that the
next pipeline stage (mesh_classification.py) really needs — and, along
the way, the DOI returned by PubMed is used to fill in the DOI of any
article that was missing one.

Quota-exhaustion handling (HTTP 429): both Scopus and PubMed limit the
number of requests they accept. As soon as either one responds with a
429, the entire process STOPS immediately (without retrying or
waiting): everything resolved up to that point is saved to output_csv,
and the console reports when the quota is expected to reset (if the
API itself reports it). In addition, every _CHECKPOINT_EVERY rows a
partial checkpoint is saved as well, in case the process were
interrupted for any other reason (manual shutdown, power outage,
unrecoverable network error...). Thanks to this, and to the resume mode
(the `resume` parameter, enabled by default), rerunning this same
script over the same output_csv picks up the work right where it left
off, without repeating requests already made.

Usage as a script (through main.py, not directly): this module is
invoked by modes 1 and 2 of main.py (--scopus-input / --resume-scopus),
for example:
    python -m scripts.main --resume-scopus dataset/test.csv \
        --scopus-api-key SCOPUS_API_KEY --pubmed-api-key PUMBED_API_KEY \
        --mesh-descriptors dataset/desc2026.xml --output-dir results/
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union
from xml.etree import ElementTree as ET
from .data_io import InputError, read_csv_clean, read_input_csv, to_csv_clean
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re as _re
import pandas as pd
import requests

# Endpoints of the three APIs used by this module:
#   - ESEARCH_URL: PubMed search engine (E-utilities). Given a search
#     term (a DOI or a quoted title), it returns a list of candidate
#     PMIDs that might match.
#   - EFETCH_URL: PubMed download endpoint (E-utilities). Given one or
#     several specific PMIDs, it returns their full XML record (title,
#     DOI, publication date, MeSH descriptors...).
#   - SCOPUS_ABSTRACT_URL: Elsevier/Scopus API for a specific ScopusID
#     ("{}" is replaced with the ScopusID itself); returns the metadata
#     Scopus has for that article, including its PMID if Scopus already
#     knows it.
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SCOPUS_ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/{}"

# Values the Status column can take when the article WAS correctly
# identified in PubMed (through any of the three routes of the cascade
# described above). Used as a set because only membership matters
# ("is this Status one of the good ones?"), never order. It serves two
# purposes: deciding, in resume mode, which rows from a previous run
# can be reused without querying Scopus again, and deciding at the end
# of convert_scopus_to_pmid which PMIDs are worth requesting PubMed
# enrichment for (the ones NOT here, i.e. "NOT_FOUND", have no PMID to
# start from).
_RESOLVED_statusES = {"SCOPUS_PMID", "DOI_MATCH", "TITLE_MATCH"}

# Internal separator for lists within a cell (MeSHList, MeSHTerms).
# "|" is used instead of ";" or "," to avoid clashing with the CSV's
# column separator, whatever it may be (some regional Excel
# configurations use ";" as the column separator when opening a CSV
# with a double click).
_LIST_SEP = "|"

# Columns contributed by the efetch enrichment in fetch_pubmed_details:
# DOI (in case the article had no known DOI from Scopus, the one from
# PubMed is used) and the two MeSH fields (MeSHList/MeSHTerms), which
# are the main reason for this second pass, because
# mesh_classification.py needs them to be able to classify the article
# into A/C/H.
_PUBMED_DETAIL_COLUMNS = ["DOI", "MeSHList", "MeSHTerms"]

# =====================================
# QUOTA EXHAUSTION / RATE LIMITING HANDLING
# =====================================
# HTTP code indicating that the API has stopped responding due to
# quota/request limits (Elsevier and NCBI use 429 for this).
_QUOTA_status_CODES = {429}

# How often (in rows) a partial save ("checkpoint") of the output CSV
# is performed, so as not to lose progress if the process is cut off.
_CHECKPOINT_EVERY = 25

# HTTP session shared by ALL requests in this module (both to Scopus
# and to PubMed), configured with automatic retries ONLY for
# transient server failures (500/502/503/504) or connection failures:
# these are errors that usually resolve themselves if retried after a
# while, so urllib3 retries them on its own (up to 5 times, with
# increasing wait 2-4-8-16-32 s) without the rest of the code having to
# worry about it. 429 (quota exhausted) is deliberately excluded from
# this automatic mechanism (it is not in status_forcelist): that case
# is handled manually in _do_request, further below, because it makes
# no sense to simply retry — the entire execution must stop, not just
# the specific request.
_SESSION = requests.Session()
_retry_strategy = Retry(
    total=5,
    backoff_factor=2,               # 2, 4, 8, 16, 32 s
    status_forcelist=[500, 502, 503, 504],  # 429 is handled separately
    allowed_methods=["GET", "POST"],
)
_adapter = HTTPAdapter(max_retries=_retry_strategy, pool_maxsize=10)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)

class QuotaExceededError(Exception):
    """Raised as soon as the API responds with 429 (quota/rate limit
    exhausted).

    It is not retried or waited on: the caller must catch it, save the
    progress made up to that point, and stop execution entirely (see
    _save_and_stop, inside convert_scopus_to_pmid). The idea is that
    Scopus/PubMed quotas usually take hours or days to reset, so there
    is no point in the script waiting or retrying in a loop: it is
    better to stop, clearly inform the user, and let them decide when
    to relaunch it.

    Attributes:
        service: name of the service that returned the 429 ("scopus"
            or "PubMed"), used only for the message printed to the
            console.
        reset_epoch: instant (Unix timestamp) at which the API says the
            quota will reset, if reported in the HTTP header
            'X-RateLimit-Reset'. May be None if the API did not include
            it in the response.
    """

    def __init__(self, service: str, message: str, reset_epoch: Optional[int] = None):
        super().__init__(message)
        self.service = service
        self.reset_epoch = reset_epoch

    def reset_human(self) -> str:
        """Converts reset_epoch to human-readable UTC text (to print to
        the console). If no reset_epoch is available, returns text
        explaining that the API did not report it, instead of failing
        or returning something empty."""
        if self.reset_epoch is None:
            return "unknown (the API did not report X-RateLimit-Reset)"
        try:
            import datetime
            dt = datetime.datetime.fromtimestamp(self.reset_epoch, tz=datetime.timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            # In case reset_epoch held a value that cannot be converted
            # to a date (unlikely, but returning the raw number is
            # preferred over raising an error here).
            return str(self.reset_epoch)

_RETRY_AFTER = 600  # 10 min: time after which a failed key can be retried

class ApiKeyPool:
    """
    Manages a list of Scopus/Elsevier API keys and rotates cyclically
    to the next one when the current one is exhausted (429).

    - Cyclic rotation: after the last key it goes back to the first.
    - A key that failed can be retried after _RETRY_AFTER seconds
      (10 min) have passed since its last failure.
    - Execution only stops (advance() -> False) when ALL keys are in
      cooldown at the same time (all have failed and none has yet
      exceeded _RETRY_AFTER seconds): there is no usable key left right
      now.

    Accepts a single key (str), several separated by commas
    ("K1,K2,K3"), or already a list/tuple of keys.
    """

    def __init__(self, api_keys: Union[str, list, tuple]):
        # Normalizes the input into a list of clean strings, accepting
        # either a single key or several separated by commas (if it
        # comes as a str) or already a list/tuple (if the caller
        # already had them separated).
        if isinstance(api_keys, str):
            keys = [k.strip() for k in api_keys.split(",")]
        else:
            keys = [str(k).strip() for k in api_keys]
        keys = [k for k in keys if k]  # discards empty entries (e.g. an extra comma)

        if not keys:
            raise ValueError("At least one Elsevier/Scopus API key is required.")

        self._keys = keys
        self._index = 0  # position of the currently active key within self._keys
        self._failed_at: dict[str, float] = {}  # key -> epoch (time.monotonic) of last failure
        self.exhausted_keys: list[str] = []      # full history, for logging only

    @property
    def current_key(self) -> str:
        """The API key that should be used RIGHT NOW for the next
        request."""
        return self._keys[self._index]

    @property
    def n_total(self) -> int:
        """Total number of keys in the pool (exhausted or not)."""
        return len(self._keys)

    def headers(self) -> dict:
        """HTTP headers expected by the Elsevier/Scopus API, with the
        currently active key."""
        return {"X-ELS-APIKey": self.current_key, "Accept": "application/json"}

    def _expire_old_failures(self, now: float) -> None:
        """Keys whose last failure is more than _RETRY_AFTER seconds
        old are considered available again."""
        for key, failed_at in list(self._failed_at.items()):
            if now - failed_at > _RETRY_AFTER:
                del self._failed_at[key]

    def advance(self) -> bool:
        """Marks the current key as failed and rotates cyclically to
        the next available key.

        Returns:
            True if there is still some key without a recent failure to
            retry with.
            False if ALL keys are in cooldown right now (all have
            failed and none has yet exceeded _RETRY_AFTER seconds):
            there is no usable key left, the caller must stop execution
            and save results instead of waiting (Scopus/PubMed keys
            usually take days to reset, so it makes no sense for the
            script to keep retrying)."""
        now = time.monotonic()
        self._failed_at[self.current_key] = now
        if self.current_key not in self.exhausted_keys:
            self.exhausted_keys.append(self.current_key)

        # Before deciding whether any usable key remains, the ones that
        # already completed their cooldown period are released: this
        # way a key that failed a long time ago can be tried again
        # instead of being discarded forever.
        self._expire_old_failures(now)

        if len(self._failed_at) == len(self._keys):
            # All keys are in cooldown at the same time: there is none
            # left to continue with, execution must stop.
            return False

        # Starting from the next position and wrapping around
        # cyclically if needed, looks for the first key that is NOT
        # currently in cooldown.
        for _ in range(len(self._keys)):
            self._index = (self._index + 1) % len(self._keys)
            if self.current_key not in self._failed_at:
                break

        return True

def _do_request(
    method: str,
    url: str,
    *,
    service: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 30,
    max_connection_retries: int = 5,
    connection_retry_backoff: float = 2.0,
) -> requests.Response:
    """Makes an HTTP request with retries on connection failures.

    If the response is 429, immediately raises QuotaExceededError,
    including the reset time if the API reports it in the
    'X-RateLimit-Reset' header. Any other Status (200, 404, etc.) is
    returned as-is for the caller to decide.

    Note on the two retry levels that coexist here: the HTTP session
    (_SESSION, defined at the top of the file) already retries codes
    500/502/503/504 on its own, at the urllib3 level. This function's
    `max_connection_retries` loop is a second, independent level,
    meant for the case where the request does not even manage to
    complete (the connection drops, there is a timeout, DNS fails,
    etc. — any requests.exceptions.RequestException), which urllib3
    does not cover because there is not even an HTTP response to work
    with there.

    Args:
        method: HTTP verb ("GET", "POST"...).
        url: full URL to call.
        service: service name ("scopus" or "PubMed"), used only for
            the messages that are printed/raised.
        headers, params: passed as-is to requests.
        timeout: maximum wait time per attempt, in seconds.
        max_connection_retries: number of retries on connection
            failures (not on HTTP error codes, which are already
            handled by the session or, in the case of 429, by this
            same function further below).
        connection_retry_backoff: base of the exponential backoff
            between retries due to connection failure (2, 4, 8, 16, 32
            seconds with the default value of 2.0).

    Returns:
        The HTTP response exactly as requests returns it, for any
        status code that is NOT 429 (including, for example, a 404,
        which each caller interprets in its own way).

    Raises:
        QuotaExceededError: if the response is 429.
        requests.exceptions.RequestException: if the connection-failure
            retries are exhausted without getting a response.
    """
    last_exc: Optional[Exception] = None
    r: Optional[requests.Response] = None

    for attempt in range(max_connection_retries + 1):
        try:
            r = _SESSION.request(method, url, headers=headers, params=params, timeout=timeout)
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < max_connection_retries:
                wait = connection_retry_backoff * (2 ** attempt)  # exponential backoff
                print(f"  [Connection to {service} interrupted ({type(exc).__name__}), "
                      f"retry {attempt + 1}/{max_connection_retries} in {wait:.1f}s...]")
                time.sleep(wait)
            else:
                # Connection retries are exhausted: the last exception
                # is propagated as-is, instead of continuing to wait.
                raise
    else:
        # This `for` loop's `else` block only runs if the loop ended
        # WITHOUT going through a `break` (that is, neither a response
        # was obtained nor was the exception re-raised above). In
        # practice this should never execute, because the `raise` on
        # the last attempt already cuts the flow before this; it is
        # left here as a safety net.
        raise last_exc

    if r.status_code in _QUOTA_status_CODES:
        # Quota exhausted: an attempt is made to read from the HTTP
        # header at what time it is expected to reset, in order to
        # report it to the console. If the header is not present, or
        # cannot be interpreted as a number, reset_epoch is left as
        # None (see QuotaExceededError.reset_human, which already
        # knows how to display that case).
        reset_epoch = None
        reset_header = r.headers.get("X-RateLimit-Reset")
        if reset_header:
            try:
                reset_epoch = int(reset_header)
            except ValueError:
                reset_epoch = None
        raise QuotaExceededError(
            service=service,
            message=f"{service} responded with 429 (quota/rate limit exhausted).",
            reset_epoch=reset_epoch,
        )

    return r

def _normalize_title(title: Optional[str]) -> str:
    """Normalizes an article title so it can be compared against another
    title (possibly from a different source, Scopus vs PubMed) without
    small formatting differences preventing them from being recognized
    as "the same" title: converts everything to lowercase, removes any
    embedded HTML tag (e.g. "<i>" in titles with italicized words),
    replaces any character that is not a letter, digit, or space with a
    space (commas, periods, colons, quotes... are all removed), and
    collapses any sequence of spaces into a single one, also trimming
    leading/trailing spaces.

    Always returns a string (never None): if `title` is None or empty,
    it returns "" directly without applying any transformation.

    Args:
        title: original title, exactly as it comes from Scopus or
            PubMed.

    Returns:
        The normalized title, ready to be compared with `==` against
        another equally normalized title.
    """
    if not title:
        return ""
    t = title.lower()
    t = _re.sub(r"<[^>]+>", " ", t)  # removes HTML tags like <i>...</i>
    t = _re.sub(r"[^\w\s]", " ", t, flags=_re.UNICODE)  # removes punctuation, keeps letters/digits/spaces
    t = _re.sub(r"\s+", " ", t)  # collapses multiple spaces into one
    return t.strip()


def _normalize_doi_for_match(doi: Optional[str]) -> str:
    """Normalizes a DOI so it can be reliably compared against another
    DOI, whatever the exact format it is written in: converts it to
    lowercase, and strips any of the usual URL/scheme prefixes a DOI is
    sometimes represented with (for example "https://doi.org/10.1000/xyz"
    and "10.1000/xyz" must normalize to the same result, because they
    are the same DOI written in two different ways).

    Always returns a string (never None): if `doi` is None or empty, it
    returns "" directly.

    Args:
        doi: original DOI, exactly as it comes from Scopus or PubMed.

    Returns:
        The normalized DOI (no prefix, lowercase, no extra spaces),
        ready to be compared with `==` against another equally
        normalized DOI.
    """
    if not doi:
        return ""
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://dx.doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _verify_pubmed_match(
    scopus_doi: Optional[str],
    scopus_title: Optional[str],
    scopus_year: Optional[str],
    pubmed_doi: Optional[str],
    pubmed_title: Optional[str],
    pubmed_year: Optional[str],
) -> bool:
    """
    Verifies whether a candidate found in PubMed is REALLY the same
    article as the one being searched for from Scopus (it is not
    enough for PubMed to have returned the candidate in a search: a
    title search, above all, can return similar but different
    articles, so this extra check is needed before accepting the PMID
    as valid). A simplified criterion is used, without comparing
    authors:

      - If both DOIs (once normalized with _normalize_doi_for_match)
        exist and match exactly: it is considered VERIFIED directly,
        without looking at title or year — a matching DOI is
        sufficient proof on its own that it is the same article.
      - If there is no DOI to compare (or it does not match), title
        AND year are tried together: only if the normalized title
        (with _normalize_title) matches EXACTLY, and the year also
        matches, is it considered verified. The year is required in
        addition to the title because the title alone, even if it
        matches exactly after normalization, is not sufficient
        guarantee (two different articles, or a review and its
        reprint in another year, could share a title).
      - Any other case (neither DOI nor title+year match): not
        verified.

    Args:
        scopus_doi, scopus_title, scopus_year: metadata from the Scopus
            side (the article being identified).
        pubmed_doi, pubmed_title, pubmed_year: metadata of the
            candidate found in PubMed, to be checked against the
            previous ones.

    Returns:
        True if it is considered the same article, False otherwise.
    """
    sdoi = _normalize_doi_for_match(scopus_doi)
    pdoi = _normalize_doi_for_match(pubmed_doi)
    if sdoi and pdoi and sdoi == pdoi:
        return True

    sTitle = _normalize_title(scopus_title)
    pTitle = _normalize_title(pubmed_title)
    syear = str(scopus_year).strip() if scopus_year else ""
    pyear = str(pubmed_year).strip() if pubmed_year else ""
    if sTitle and pTitle and sTitle == pTitle and syear and pyear and syear == pyear:
        return True

    return False

# =====================================
# SCOPUS -> PMID
# =====================================
def _get_scopus_metadata(ScopusID: str, key_pool: ApiKeyPool, timeout: int = 30) -> Optional[dict]:
    """Queries the Scopus API (Abstract Retrieval) for a specific
    ScopusID and returns its basic metadata.

    If the current key returns 429 (quota exhausted), it automatically
    rotates to the next available key in the pool (see
    ApiKeyPool.advance) and retries the SAME request with the new key,
    in a `while True` loop that only ends when a non-429 response is
    obtained or when no key is left available (in which case
    ApiKeyPool.advance() returns False and the QuotaExceededError is
    re-raised to the caller).

    Args:
        ScopusID: Scopus identifier of the article to query.
        key_pool: pool of Elsevier/Scopus API keys (see ApiKeyPool).
        timeout: maximum wait time for the HTTP request, in seconds.

    Returns:
        A dictionary {"doi", "title", "pmid", "year"} with what Scopus
        knows about the article, or None if Scopus responds 404 (the
        ScopusID does not exist) or any other error code that is not
        429 (treated the same as "not found", so as not to block the
        pipeline over an unexpected code, e.g. a 400 from a malformed
        ScopusID) — or if the response cannot be interpreted as JSON.

    Raises:
        QuotaExceededError: if ALL keys in the pool are exhausted
            without obtaining a valid response.
    """
    url = SCOPUS_ABSTRACT_URL.format(ScopusID)
    while True:
        try:
            r = _do_request("GET", url, service="scopus", headers=key_pool.headers(), timeout=timeout)
            break
        except QuotaExceededError:
            if not key_pool.advance():
                raise
            print(f"  [API key exhausted, rotating to the next one ({len(key_pool.exhausted_keys)}/{key_pool.n_total} exhausted)]")

    if r.status_code == 404:
        return None
    if r.status_code != 200:
        # any other unhandled code: treated as "not found" so as not to
        # block the pipeline over unusual codes not related to
        # throttling (e.g. 400 from a malformed ScopusID).
        return None

    try:
        data = r.json()
    except ValueError:
        return None

    # Structure of the Scopus Abstract Retrieval response: the useful
    # content is nested inside "abstracts-retrieval-response" ->
    # "coredata". prism:coverDate is the publication date in
    # "YYYY-MM-DD" format (or similar); only the first 4 characters are
    # taken as the year, after checking they are actually digits (if
    # coverDate came empty or with an unexpected format, year is left
    # as None instead of storing garbage).
    item = data.get("abstracts-retrieval-response", {})
    core = item.get("coredata", {})
    cover_date = core.get("prism:coverDate")
    year = cover_date[:4] if cover_date and cover_date[:4].isdigit() else None
    return {
        "doi": core.get("prism:doi"),
        "title": core.get("dc:title"),
        # Note: "pubmed-id" is nested inside "coredata", not at the top
        # level of "abstracts-retrieval-response".
        "pmid": core.get("pubmed-id"),
        "year": year,
    }

def _pmid_from_doi(
    doi: Optional[str], scopus_title: Optional[str] = None,
    scopus_year: Optional[str] = None, pubmed_api_key: Optional[str] = None, timeout: int = 30,
) -> tuple[Optional[str], Optional[str]]:
    """Tries to find an article's PMID by searching its DOI in PubMed
    (second route of the resolution cascade, when Scopus did not bring
    the PMID directly but did bring a DOI).

    Args:
        doi: article's DOI (comes from Scopus). If it is None, (None,
            None) is returned without making any request.
        scopus_title, scopus_year: additional Scopus metadata, used
            only for verifying the candidate (see
            _verify_pubmed_match), not for the search itself.
        pubmed_api_key, timeout: forwarded as-is to
            _search_and_verify_pmid.

    Returns:
        Tuple (pmid, year): pmid is the PMID found and verified, or
        None if none was found; year is the year PubMed provides for
        that same already-verified candidate (useful as a fallback
        when Scopus itself did not give a year for the article), or
        None if there was no match.
    """
    if doi is None:
        return None, None
    return _search_and_verify_pmid(f"{doi}[DOI]", doi, scopus_title, scopus_year, pubmed_api_key=pubmed_api_key, timeout=timeout)

def _pmid_from_title(
    Title: Optional[str], scopus_doi: Optional[str] = None,
    scopus_year: Optional[str] = None, pubmed_api_key: Optional[str] = None, timeout: int = 30,
) -> tuple[Optional[str], Optional[str]]:
    """Tries to find an article's PMID by searching its title in PubMed
    (third and last route of the resolution cascade, when neither
    Scopus's direct PMID nor the DOI search worked).

    Args:
        Title: article's title (comes from Scopus). If it is None or
            empty, (None, None) is returned without making any
            request.
        scopus_doi, scopus_year: additional Scopus metadata, used only
            for verifying the candidate (see _verify_pubmed_match), not
            for the search itself.
        pubmed_api_key, timeout: forwarded as-is to
            _search_and_verify_pmid.

    Returns:
        Tuple (pmid, year), with the same meaning as in
        _pmid_from_doi.
    """
    if not Title:
        return None, None
    return _search_and_verify_pmid(f'"{Title}"[Title]', scopus_doi, Title, scopus_year, pubmed_api_key=pubmed_api_key, timeout=timeout)

def _search_and_verify_pmid(
    term: str,
    scopus_doi: Optional[str],
    scopus_title: Optional[str],
    scopus_year: Optional[str],
    pubmed_api_key: Optional[str] = None,
    retmax: int = 25,
    timeout: int = 30,
) -> tuple[Optional[str], Optional[str]]:
    """
    Function shared by _pmid_from_doi and _pmid_from_title: searches
    esearch (PubMed's search engine) with the search term `term`
    (which may be a DOI or title search, depending on who calls it),
    and verifies EACH candidate returned by the search, one by one and
    in the order PubMed returns them, downloading its full detail with
    efetch and checking with _verify_pubmed_match whether it really is
    the same article as the one from Scopus. As soon as it finds the
    first candidate that verifies, it stops right there (early exit)
    and returns it, without checking the rest of the candidates even
    if more remained in the list.

    Returns (pmid, year): year is the year PubMed provides for that
    already-verified candidate, so it can be used as a fallback when
    Scopus does not give a year for the article.

    Args:
        term: search term already formatted for esearch (e.g.
            '10.1000/xyz[DOI]' or '"Article title"[Title]').
        scopus_doi, scopus_title, scopus_year: Scopus metadata of the
            article being searched for, needed to verify each
            candidate.
        pubmed_api_key: NCBI/PubMed E-utilities API key (optional).
        retmax: maximum number of candidates to request from esearch.
        timeout: maximum wait time for each HTTP request.

    Returns:
        (None, None) if esearch does not respond 200, if the response
        cannot be interpreted as XML, if there is no candidate at all,
        or if none of the candidates verify; otherwise, (pmid, year)
        of the first verified candidate.
    """
    params = {"db": "pubmed", "retmode": "xml", "term": term, "retmax": str(retmax)}
    if pubmed_api_key:
        params["api_key"] = pubmed_api_key
    r = _do_request("GET", ESEARCH_URL, service="PubMed", params=params, timeout=timeout)
    if r.status_code != 200:
        return None, None
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None, None
    # esearch returns, inside <IdList>, one <Id> per candidate PMID
    # that matches the search term (not yet verified as really being
    # the same article).
    candidate_pmids = [el.text for el in root.findall(".//Id") if el.text]
    if not candidate_pmids:
        return None, None

    for pmid in candidate_pmids:
        # The full detail of this candidate is requested (one efetch
        # call per candidate, not in bulk, precisely because as soon as
        # one verifies there is no need to keep requesting the rest).
        details = _fetch_pubmed_batch([pmid], pubmed_api_key=pubmed_api_key, timeout=timeout)
        detail = details.get(pmid)
        if detail is None:
            continue
        if _verify_pubmed_match(
            scopus_doi, scopus_title, scopus_year,
            detail.get("DOI"), detail.get("Title"), detail.get("Year"),
        ):
            return pmid, detail.get("Year")
    return None, None

def _resolve_one(ScopusID: str, key_pool: ApiKeyPool, pubmed_api_key: Optional[str] = None) -> dict:
    """Resolves a single ScopusID by going through the full cascade
    described at the top of the file: first Scopus, and if needed,
    PubMed by DOI and by title, in that order, stopping as soon as a
    route produces a result.

    It also tries to fill in the publication year (`year`) when Scopus
    does not provide it, taking advantage of any of the PubMed queries
    already made as part of the cascade itself:
      - If the PMID came directly from Scopus (Status "SCOPUS_PMID")
        but without a year, an extra query is made to PubMed (efetch)
        with that same PMID solely to recover the year, without needing
        any additional search or verification (it is already known
        with certainty that it is the same article).
      - If the PMID was found by DOI or by title (Status "DOI_MATCH" or
        "TITLE_MATCH"), the year of the already-verified candidate
        returned by _search_and_verify_pmid is reused directly as
        year, without any additional query.
      - If Scopus already had a year, it is never overwritten with
        PubMed's (it is only filled in when it was missing).

    Immediately raises QuotaExceededError if any of the calls receives
    a 429 (quota exhausted); the caller must catch it, save progress,
    and stop execution.

    Args:
        ScopusID: Scopus identifier of the article to resolve.
        key_pool: pool of Elsevier/Scopus API keys.
        pubmed_api_key: NCBI/PubMed E-utilities API key (optional).

    Returns:
        A dictionary with the keys PMID, DOI, Title, Year and Status.
        If Scopus does not recognize the ScopusID at all
        (_get_scopus_metadata returns None), it is returned with all
        fields empty and Status="NOT_FOUND".
    """
    pmid, doi, title, year, status = None, None, None, None, "NOT_FOUND"

    meta = _get_scopus_metadata(ScopusID, key_pool)
    if meta:
        doi = meta["doi"]
        title = meta["title"]
        year = meta.get("year")

        if meta["pmid"]:
            pmid = meta["pmid"]
            status = "SCOPUS_PMID"
            if year is None:
                # Scopus gave the PMID directly but not the year: that
                # same PMID is queried in PubMed solely to recover the
                # year, without needing any search/verification.
                pubmed_detail = _fetch_pubmed_batch([pmid], pubmed_api_key=pubmed_api_key).get(pmid)
                if pubmed_detail:
                    year = pubmed_detail.get("Year")

        if pmid is None and doi:
            pmid, pubmed_year = _pmid_from_doi(doi, scopus_title=title, scopus_year=year, pubmed_api_key=pubmed_api_key)
            if pmid:
                status = "DOI_MATCH"
                if year is None:
                    year = pubmed_year

        if pmid is None and title:
            pmid, pubmed_year = _pmid_from_title(title, scopus_doi=doi, scopus_year=year, pubmed_api_key=pubmed_api_key)
            if pmid:
                status = "TITLE_MATCH"
                if year is None:
                    year = pubmed_year

    return {"PMID": pmid, "DOI": doi, "Title": title, "Year": year, "Status": status}


# =====================================
# PMID -> PUBMED METADATA (efetch)
# =====================================
def _parse_pubmed_article(article_elem: ET.Element) -> Optional[dict]:
    """Extracts, from a single <PubmedArticle> node in the XML returned
    by efetch, all the fields this pipeline needs: PMID, DOI, title,
    publication year, and the article's MeSH descriptors.

    Args:
        article_elem: already-parsed <PubmedArticle> node (an
            xml.etree.ElementTree element).

    Returns:
        None if the node does not even carry a PMID (a degenerate case
        that should not happen in practice, but is checked for safety).
        Otherwise, a dictionary with the keys:
          - PMID: PubMed identifier (mandatory).
          - DOI: article's DOI, looked up among the
            <ArticleId IdType="doi"> entries, or None if none appears.
          - Title: full article title. `itertext()` is used instead of
            reading `.text` directly because the title may come with
            nested formatting tags (e.g. "<i>" for italicized text),
            and itertext() walks and concatenates all the text, whether
            or not it is inside sub-tags.
          - Year: publication year. First looked up in the structured
            <year> field; if absent (some old PubMed records only carry
            a free-text date inside <MedlineDate>, e.g. "2020 Jan-Feb"),
            an attempt is made to extract a 4-digit year (starting with
            19 or 20) from that text with a regular expression.
          - MeSHList: UI codes of the article's MeSH descriptors, joined
            with the internal separator "|" (see _LIST_SEP), or None if
            the article has no associated MeSH term.
          - MeSHTerms: human-readable names of those same descriptors,
            in the same order as MeSHList (position by position), also
            joined with "|".
    """
    pmid_elem = article_elem.find(".//MedlineCitation/PMID")
    if pmid_elem is None or not pmid_elem.text:
        return None
    pmid = pmid_elem.text.strip()

    doi = None
    for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi" and article_id.text:
            doi = article_id.text.strip()
            break

    title_elem = article_elem.find(".//MedlineCitation/Article/ArticleTitle")
    Title = "".join(title_elem.itertext()).strip() if title_elem is not None else None

    year = None
    year_elem = article_elem.find(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/year")
    if year_elem is not None and year_elem.text:
        year = year_elem.text.strip()
    else:
        # Fallback for PubMed records that do not carry the year in the
        # structured <year> field, only inside a free-text field like
        # "2020 Jan-Feb" in <MedlineDate>: the first 4-digit group
        # starting with 19 or 20 is searched for there.
        medline_date = article_elem.find(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date is not None and medline_date.text:
            m = _re.search(r"(19|20)\d{2}", medline_date.text)
            if m:
                year = m.group(0)

    # MeSH terms: stored both as a list of UI codes (for
    # classify_mesh_codes) and as human-readable text, separated by "|"
    # so as not to clash with the CSV's column separator (neither ","
    # nor ";").
    mesh_uis = []
    mesh_terms = []
    for heading in article_elem.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
        ui = heading.attrib.get("UI")
        term = heading.text.strip() if heading.text else None
        if ui:
            mesh_uis.append(ui)
        if term:
            mesh_terms.append(term)

    return {
        "PMID": pmid,
        "DOI": doi,
        "Title": Title,
        "Year": year,
        "MeSHList": _LIST_SEP.join(mesh_uis) if mesh_uis else None,
        "MeSHTerms": _LIST_SEP.join(mesh_terms) if mesh_terms else None,
    }


def _fetch_pubmed_batch(pmids: list[str], pubmed_api_key: Optional[str] = None, timeout: int = 60) -> dict[str, dict]:
    """Calls efetch ONCE to request the full detail of a batch of PMIDs
    at the same time (more efficient than one call per PMID), and
    returns the already-parsed results, indexed by PMID.

    Immediately raises QuotaExceededError if PubMed returns 429 (unlike
    _get_scopus_metadata, here it does NOT retry by rotating keys:
    PubMed does not use a key pool like Scopus, so a 429 here is
    propagated directly upward).

    Args:
        pmids: list of PMIDs to query in this call. If empty, {} is
            returned directly without making any HTTP request.
        pubmed_api_key: NCBI/PubMed E-utilities API key (optional).
        timeout: maximum wait time for the HTTP request.

    Returns:
        Dictionary {pmid: detail}, where each `detail` is the
        dictionary returned by _parse_pubmed_article for that article.
        If the response is not 200, or cannot be parsed as XML, an
        empty dictionary is returned instead of raising an error
        (batch efetch is treated as "best effort": if a whole batch
        fails, those PMIDs simply end up without detail in this pass).
    """
    if not pmids:
        return {}
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if pubmed_api_key:
        params["api_key"] = pubmed_api_key
    r = _do_request("GET", EFETCH_URL, service="PubMed", params=params, timeout=timeout)
    if r.status_code != 200:
        return {}
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return {}

    details_by_pmid: dict[str, dict] = {}
    for article in root.findall(".//PubmedArticle"):
        parsed = _parse_pubmed_article(article)
        if parsed:
            details_by_pmid[parsed["PMID"]] = parsed
    return details_by_pmid


def fetch_pubmed_details(
    pmids: list[str],
    pubmed_api_key: Optional[str] = None,
    batch_size: int = 200,
    sleep_time: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Downloads PubMed metadata (mainly MeSHList/MeSHTerms, plus DOI as a
    fallback) for an ENTIRE list of PMIDs, splitting it into batches of
    `batch_size` so as not to send a single giant request to efetch,
    and respecting a pause between batches to stay under NCBI's
    requests-per-second limit.

    If the API returns 429 (quota exhausted) on some batch, it stops
    immediately: it does not retry or wait. QuotaExceededError is
    propagated upward, leaving in the `partial_details` attribute (dict
    {pmid: details}) everything obtained in the previous batches that
    did work, so the caller can save it instead of losing that
    progress.

    Args:
        pmids: list of PMIDs (strings) to query. They are normalized
            (spaces stripped, empty entries and the literal text "nan"
            discarded) and deduplicated with a `set` before splitting
            into batches, so it does not matter if the input list
            carries repeated PMIDs.
        pubmed_api_key: NCBI/PubMed E-utilities API key. It is
            OPTIONAL: without it, E-utilities still works the same,
            just limited to 3 requests/second per IP instead of 10 with
            a key.
        batch_size: number of PMIDs per efetch call (recommended <=
            200).
        sleep_time: pause between batches (seconds), to respect NCBI's
            limits. If not specified (None), it is computed
            automatically depending on whether there is a
            pubmed_api_key: 0.3 s with a key (~10 req/s), 0.5 s without
            one (~2 req/s, with margin under the actual 3 req/s limit).
        verbose: parameter accepted for compatibility with the rest of
            the module's functions, although this particular function
            does not print anything to the console directly (progress
            is printed, if at all, by the code that calls it).

    Returns:
        DataFrame indexed by PMID, with columns DOI/MeSHList/MeSHTerms
        (those in _PUBMED_DETAIL_COLUMNS). If `pmids` ends up empty
        after normalization, an empty DataFrame is returned but already
        with those columns defined, so the code that uses it afterward
        does not have to check beforehand whether it is empty.

    Raises:
        QuotaExceededError: if PubMed responds 429 on some batch. The
            dict of details obtained up to that point is attached as
            `error.partial_details`.
    """
    if sleep_time is None:
        sleep_time = 0.3 if pubmed_api_key else 0.5
    unique_pmids = sorted({str(p).strip() for p in pmids if p and str(p).strip().lower() != "nan"})
    if not unique_pmids:
        return pd.DataFrame(columns=["PMID"] + _PUBMED_DETAIL_COLUMNS).set_index("PMID")

    all_details: dict[str, dict] = {}
    n_batches = (len(unique_pmids) + batch_size - 1) // batch_size  # round up

    for b in range(n_batches):
        batch = unique_pmids[b * batch_size:(b + 1) * batch_size]
        try:
            batch_details = _fetch_pubmed_batch(batch, pubmed_api_key=pubmed_api_key)
        except QuotaExceededError as exc:
            # What was obtained in previous batches is attached as a
            # dynamic attribute of the exception itself, so that
            # fetch_pubmed_details (and, further up,
            # convert_scopus_to_pmid) can recover it without having to
            # return it through another channel.
            exc.partial_details = all_details  # type: ignore[attr-defined]
            raise

        all_details.update(batch_details)
        time.sleep(sleep_time)

    rows = []
    for pmid in unique_pmids:
        # If a specific PMID does not appear in all_details (for
        # example because efetch found no record for it, or that batch
        # failed for a reason unrelated to the quota), a row with all
        # fields empty is added for it, instead of omitting it: this
        # way the output DataFrame always has one row per requested
        # PMID.
        detail = all_details.get(pmid, {})
        rows.append({
            "PMID": pmid,
            "DOI": detail.get("DOI"),
            "MeSHList": detail.get("MeSHList"),
            "MeSHTerms": detail.get("MeSHTerms"),
        })

    return pd.DataFrame(rows).set_index("PMID")


# =====================================
# MAIN PIPELINE
# =====================================
def convert_scopus_to_pmid(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    api_keys: Union[str, list, tuple],
    pubmed_api_key: Optional[str] = None,
    sleep_time: float = 0.2,
    verbose: bool = True,
    resume: bool =  True,
    pubmed_batch_size: int = 200,
    pubmed_sleep_time: Optional[float] = None,
    author_id_column: str = "AuthorId",
    scopus_id_column: str = "ScopusID",
    year_column:  str = "Year",
) -> pd.DataFrame:
    """
    Main function of the module: runs stage 1 of the pipeline
    completely, from start to finish.

    Steps, in order:
      1. Reads `input_csv` (with read_input_csv, which detects on its
         own whether the separator is ';' or ','), locates the
         ScopusID, researcher, and year columns (with the names given
         in scopus_id_column/author_id_column/year_column), renames
         them to 'ScopusID'/'AuthorID'/'Year' (the names the rest of
         the pipeline uses), and discards any other column the input
         CSV carried.
      2. If `resume=True` and `output_csv` already exists from a
         previous run, loads from it the ScopusIDs that were already
         resolved (Status in SCOPUS_PMID/DOI_MATCH/TITLE_MATCH) and the
         PubMed metadata that had already been downloaded for their
         PMIDs, so as not to have to query any of that again.
      3. For each row of the input CSV, in order: if its ScopusID was
         already resolved before (step 2), that result is reused
         without making any request; otherwise, it is resolved by
         calling _resolve_one (the Scopus -> DOI -> title cascade
         described at the top of the file). Every _CHECKPOINT_EVERY
         NEW rows (not reused), a partial checkpoint is saved to
         output_csv.
      4. Once all rows are resolved, the PubMed metadata — mainly
         MeSHList/MeSHTerms — is downloaded in bulk (via
         fetch_pubmed_details) for all the PMIDs that were found and
         that had not already been reused from step 2. With that
         metadata, the DOI of any row that was missing one is also
         filled in (using the DOI PubMed provides for its PMID).
      5. The final result is saved to output_csv and returned as a
         DataFrame.

    Quota-exhaustion handling (HTTP 429):
        If Scopus or PubMed respond with a 429 (quota or rate limit
        exhausted) at any point during steps 3 or 4, the process stops
        immediately, WITHOUT retrying and WITHOUT waiting. A CSV is
        saved with everything processed up to that point and the
        console reports the quota reset time (if the API provides it
        in the 'X-RateLimit-Reset' header). Calling this same function
        again on the same output_csv (with resume=True, the default)
        picks up the work right where it left off.

    Args:
        input_csv: path to the input CSV. It may be separated by ';' or
            by ',' (detected automatically, see data_io.read_input_csv);
            from here on the whole pipeline is unified to ';'.
        output_csv: path where the resulting CSV will be saved (and, if
            it exists, read from).
        api_keys: one or several Elsevier/Scopus API keys (see
            ApiKeyPool, above, for the accepted format and rotation
            among several).
        pubmed_api_key: NCBI/PubMed E-utilities API key (optional).
        sleep_time: pause between requests to Scopus/esearch (seconds).
        verbose: if True, prints progress to the console (one line per
            article, checkpoint notices, final summary...).
        resume: if True and `output_csv` already exists, reuses the
            records whose ScopusID was already resolved (Status in
            SCOPUS_PMID/DOI_MATCH/TITLE_MATCH) and does not call Scopus
            again for them. Rows with Status="API_ERROR" or "NOT_FOUND"
            ARE retried. It also reuses PubMed metadata already
            downloaded (if the row already had a MeSHList) and only
            queries efetch for new PMIDs.
        pubmed_batch_size: number of PMIDs per efetch call.
        pubmed_sleep_time: pause between efetch batches (seconds). If
            not specified (None), it is computed automatically
            depending on whether there is a pubmed_api_key: 0.3 s with
            a key (~10 req/s), 0.5 s without one (~2 req/s, with margin
            under NCBI's actual 3 req/s limit).
        author_id_column: name, in `input_csv`, of the column with the
            researcher's identifier. Renamed to 'AuthorID'.
        scopus_id_column: name, in `input_csv`, of the column with each
            article's ScopusID. Renamed to 'ScopusID'.
        year_column: name, in `input_csv`, of the column with the
            publication year. Renamed to 'Year'. Any column of
            `input_csv` that is neither this one nor author_id_column
            nor scopus_id_column is discarded before resolution begins.

    Returns:
        DataFrame with the columns: AuthorID, ScopusID, Year, PMID,
        DOI, Title, Status, MeSHList, MeSHTerms.

        Important note on the result's Year column: the value left in
        this column is ALWAYS the one that comes from the input CSV
        (through `row["Year"]`, further down in the code), not the one
        _resolve_one internally computes — that "fallback" year
        _resolve_one computes is used only during the verification of
        PubMed candidates itself (_verify_pubmed_match needs to compare
        years), but it never ends up replacing the final Year column.
    """
    if verbose:
        print(f"Resolving ScopusID -> PMID/DOI/MeSH from '{input_csv}' (output: '{output_csv}')")

    if pubmed_sleep_time is None:
        pubmed_sleep_time = 0.3 if pubmed_api_key else 0.5
    key_pool = ApiKeyPool(api_keys)

    # The input CSV is the user's original export: it may come with ';'
    # or ',' as separator, so it is read with read_input_csv (see
    # data_io.py). From here on, everything this script saves/rereads
    # (output_csv) is unified to ';' via to_csv_clean/read_csv_clean.
    df = read_input_csv(input_csv)

    # The user can name their ScopusID, researcher, and year columns
    # however they like: they are renamed here to the names the rest
    # of the pipeline uses (ScopusID/AuthorID/Year) and any other
    # column the CSV carried is discarded, so it does not interfere
    # further along.
    missing = [c for c in (scopus_id_column, author_id_column, year_column) if c not in df.columns]
    if missing:
        # An error message aimed at the END USER (not for debugging
        # code) is assembled: it states exactly which column(s) are
        # missing, which ones ARE present in their CSV, and how to fix
        # it (the corresponding command-line arguments). It is raised
        # as InputError, not as a generic error, so that main.py can
        # catch it and display it without a traceback (see
        # data_io.InputError).
        hints = []
        if scopus_id_column in missing:
            hints.append(f"'{scopus_id_column}' (Scopus ID column for each article)")
        if author_id_column in missing:
            hints.append(f"'{author_id_column}' (researcher column)")
        if year_column in missing:
            hints.append(f"'{year_column}' (publication year column)")
        raise InputError(
            f"Column {' or column '.join(hints)} not found in '{input_csv}'.\n"
            f"Columns actually present in that CSV: {list(df.columns)}.\n"
            f"If your CSV uses different names, specify the real names with "
            f"--scopus-id-column/--author-id-column/--year_column when running main.py."
        )
    df = df.rename(columns={scopus_id_column: "ScopusID", author_id_column: "AuthorID", year_column: "Year"})
    df = df[["AuthorID", "ScopusID", "Year"]]
    df["ScopusID"] = df["ScopusID"].astype(str).str.strip()

    # --- Resume mode: load previous results if they exist ---
    # previous_by_id: ScopusID -> result already resolved in a previous
    # run (only the ones that ended up with a "good" Status, see
    # _RESOLVED_statusES). previous_pubmed_by_pmid: PMID -> PubMed
    # metadata (MeSHList/MeSHTerms) already downloaded, so as not to
    # request efetch again for PMIDs already queried before.
    previous_by_id: dict[str, dict] = {}
    previous_pubmed_by_pmid: dict[str, dict] = {}
    output_path = Path(output_csv)
    if resume and output_path.exists():
        try:
            prev_df = read_csv_clean(output_path)
            prev_df["ScopusID"] = prev_df["ScopusID"].astype(str).str.strip()
            for _, prow in prev_df.iterrows():
                if prow.get("Status") in _RESOLVED_statusES:
                    prow_year = prow.get("Year")
                    previous_by_id[prow["ScopusID"]] = {
                        "PMID": prow.get("PMID"),
                        "DOI": prow.get("DOI"),
                        "Title": prow.get("Title"),
                        # Forced to text (or None) so the type is always
                        # the same as that of a row resolved again in
                        # this same run (which also produces text),
                        # regardless of what specific type pandas gave
                        # it when rereading the previous CSV.
                        "Year": str(prow_year) if pd.notna(prow_year) else None,
                        "Status": prow.get("Status"),
                    }
                    pmid = prow.get("PMID")
                    if pmid and pd.notna(prow.get("MeSHList")):
                        previous_pubmed_by_pmid[str(pmid).strip()] = {
                            "MeSHList": prow.get("MeSHList"),
                            "MeSHTerms": prow.get("MeSHTerms"),
                        }
            if verbose and previous_by_id:
                print(f"Resuming: {len(previous_by_id)} ScopusID already resolved, skipping them.")
        except Exception:
            # If the previous output_csv could not be read (corrupt
            # file, unexpected format, etc.), it is simply ignored and
            # everything is processed from scratch, instead of letting
            # the read error interrupt the whole run.
            if verbose:
                print("Notice: could not read the previous output_csv, processing everything from scratch.")

    def _save_and_stop(partial_results: list[dict], exc: QuotaExceededError) -> pd.DataFrame:
        """Saves what has been processed so far and stops execution
        (no retries, no automatic resume). The user decides when and
        how to relaunch the script later on."""
        partial_df = pd.DataFrame(partial_results)
        to_csv_clean(partial_df, output_csv)
        print()
        print(f"[STOPPED] {exc.service} quota exhausted (HTTP 429).")
        print(f"         Reset reported by the API: {exc.reset_human()}")
        print(f"         Progress saved: {len(partial_results)}/{len(df)} rows in '{output_csv}'.")
        print("         Execution stopped. Rerun the script whenever you want to continue.")
        return partial_df

    # --- Step 1: ScopusID -> PMID/DOI/Title ---
    results = []
    n = len(df)
    new_rows_since_checkpoint = 0  # counts only rows with NEW info (not reused)

    for i, row in df.iterrows():
        ScopusID = row["ScopusID"]

        cached = previous_by_id.get(ScopusID)
        if cached is not None:
            # This ScopusID was already resolved in a previous run: it
            # is reused as-is, without spending any request to
            # Scopus/PubMed or advancing new_rows_since_checkpoint
            # (which counts only NEW work).
            if verbose:
                print(f"[{i + 1}/{n}] {ScopusID} (reused, {cached['Status']})")
            resolved = cached
        else:
            if verbose:
                print(f"[{i + 1}/{n}] {ScopusID}")

            try:
                resolved = _resolve_one(ScopusID, key_pool, pubmed_api_key=pubmed_api_key)
            except QuotaExceededError as exc:
                # Everything obtained up to BEFORE this row is saved
                # (results does not yet include the current row) and
                # execution stops immediately.
                return _save_and_stop(results, exc)

            time.sleep(sleep_time)
            new_rows_since_checkpoint += 1

        results.append({
            "AuthorID": row["AuthorID"],
            "ScopusID": ScopusID,
            # The year left here is the one from the input CSV
            # (row["Year"]), NOT the one _resolve_one computed — see
            # the note about this in this function's docstring, above
            # (the "Returns" section).
            "Year": row["Year"],
            "PMID": resolved.get("PMID"),
            "DOI": resolved.get("DOI"),
            "Title": resolved.get("Title"),
            "Status": resolved.get("Status"),
        })

        if new_rows_since_checkpoint >= _CHECKPOINT_EVERY:
            # Periodic partial save: output_csv is rewritten with
            # EVERYTHING accumulated in `results` so far (not just what
            # is new since the last checkpoint), so the file on disk is
            # always complete and usable if the process is cut off
            # right afterward.
            results_df = pd.DataFrame(results)
            to_csv_clean(results_df, output_csv)
            if verbose:
                print(f"    (checkpoint saved: {i + 1}/{n} rows processed in {output_csv})")
            new_rows_since_checkpoint = 0

    out = pd.DataFrame(results)

    # --- Step 2: bulk enrichment with PubMed metadata ---
    # It only makes sense to request PubMed metadata for the PMIDs that
    # belong to rows with a "good" Status (NOT_FOUND rows have no PMID
    # to start from). Of those, the ones already reused from a previous
    # run are also discarded (previous_pubmed_by_pmid), so as not to
    # request them again.
    pmids_to_check = out.loc[out["Status"].isin(_RESOLVED_statusES), "PMID"].dropna().astype(str).tolist()
    pmids_needed = [p for p in pmids_to_check if p not in previous_pubmed_by_pmid]

    try:
        new_details = fetch_pubmed_details(
            pmids_needed,
            pubmed_api_key=pubmed_api_key,
            batch_size=pubmed_batch_size,
            sleep_time=pubmed_sleep_time,
            verbose=verbose,
        )
        # The reused details (previous_pubmed_by_pmid) are combined with
        # the newly downloaded ones (new_details): if the same PMID
        # were in both (it should not, because pmids_needed already
        # excludes the reused ones), the most recent one wins by the
        # dictionary order in the `**` merge (the second overwrites the
        # first).
        details_by_pmid = {**previous_pubmed_by_pmid, **new_details.to_dict(orient="index")}
    except QuotaExceededError as exc:
        # PubMed's quota ran out during enrichment: whatever had been
        # downloaded from this batch (exc.partial_details) is rescued
        # and combined with what was reused, so as to be able to fill
        # in `out` with everything that WAS obtained before stopping.
        partial = getattr(exc, "partial_details", {})
        details_by_pmid = {**previous_pubmed_by_pmid, **partial}

        out["DOI"] = out["DOI"].where(
            out["DOI"].notna() & (out["DOI"].astype(str).str.strip() != ""),
            out["PMID"].astype(str).map(lambda p: details_by_pmid.get(p, {}).get("DOI")),
        )
        for col in ("MeSHList", "MeSHTerms"):
            out[col] = out["PMID"].astype(str).map(lambda p: details_by_pmid.get(p, {}).get(col))

        return _save_and_stop(out.to_dict(orient="records"), exc)

    # Normal path (quota not exhausted): the DOI PubMed provides is
    # used ONLY for rows that were missing one (out["DOI"].where(~mask,
    # ...) leaves the existing DOI untouched and only substitutes where
    # the `doi_missing_mask` mask is True), never to overwrite a DOI
    # Scopus had already given.
    pubmed_doi = out["PMID"].astype(str).map(lambda p: details_by_pmid.get(p, {}).get("DOI"))
    doi_missing_mask = out["DOI"].isna() | (out["DOI"].astype(str).str.strip() == "")
    n_doi_filled = int((doi_missing_mask & pubmed_doi.notna()).sum())
    out["DOI"] = out["DOI"].where(~doi_missing_mask, pubmed_doi)
    if verbose and n_doi_filled:
        print(f"DOI filled in from PubMed: {n_doi_filled}")

    for col in ("MeSHList", "MeSHTerms"):
        out[col] = out["PMID"].astype(str).map(
            lambda p: details_by_pmid.get(p, {}).get(col)
        )

    to_csv_clean(out, output_csv)

    if verbose:
        n_resolved = out["Status"].isin(_RESOLVED_statusES).sum()
        print()
        print("Scopus -> PMID conversion DONE")
        print(f"Found in PubMed: {n_resolved}/{len(out)}")
        n_with_mesh = out["MeSHList"].notna().sum()
        print(f"With MeSHList: {n_with_mesh}/{len(out)}")
        print(f"Result saved to {output_csv}")

    return out
