"""Standalone reference validation logic reused by CLI tools."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import uuid
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import bibtexparser
from bibtexparser.bparser import BibTexParser
from rapidfuzz import fuzz

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None

try:
    from packaging import version as packaging_version
except ImportError:  # pragma: no cover - optional dependency
    packaging_version = None

try:  # Optional dependency – fallback to CrossRef only when unavailable
    from scholarly import scholarly, ProxyGenerator
except ImportError:  # pragma: no cover - graceful degradation
    scholarly = None
    ProxyGenerator = None


logger = logging.getLogger(__name__)


_crossref_request_count = 0
_crossref_rate_lock: Optional[asyncio.Lock] = None


async def _throttle_crossref_requests() -> None:
    """Pause for 10 seconds after every 10 CrossRef lookups to avoid rate limits."""
    global _crossref_request_count, _crossref_rate_lock

    if _crossref_rate_lock is None:
        _crossref_rate_lock = asyncio.Lock()

    async with _crossref_rate_lock:
        if _crossref_request_count and _crossref_request_count % 10 == 0:
            logger.info(
                "CrossRef lookups reached %s; pausing for 10 seconds to avoid being blocked",
                _crossref_request_count,
            )
            await asyncio.sleep(10)
        _crossref_request_count += 1


def _create_ssl_context() -> Optional[ssl.SSLContext]:
    """Return an SSL context backed by certifi's CA bundle when available."""
    if certifi is None:
        return None
    try:
        return ssl.create_default_context(cafile=certifi.where())
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Falling back to default SSL context: %s", exc)
        return None


def _configure_scholarly_proxy() -> None:
    """Configure scholarly's ProxyGenerator based on environment variables."""
    if scholarly is None or ProxyGenerator is None:
        return

    mode = os.environ.get("SCHOLARLY_PROXY_MODE")
    if not mode:
        return

    mode = mode.strip().lower()
    pg = ProxyGenerator()
    configured = False

    try:
        if mode == "free":
            if httpx is None:
                logger.warning("httpx not installed; cannot enable Free proxy mode")
            else:
                hv = getattr(httpx, "__version__", "0")
                if packaging_version and packaging_version.parse(hv) < packaging_version.parse("0.24.0"):
                    logger.warning(
                        "httpx %s detected; scholarly FreeProxies requires httpx>=0.24.0. "
                        "Upgrade httpx or choose another SCHOLARLY_PROXY_MODE.",
                        hv,
                    )
                else:
                    configured = pg.FreeProxies()
        elif mode == "serpapi":
            api_key = os.environ.get("SCHOLARLY_SERP_API_KEY")
            if not api_key:
                logger.warning("SCHOLARLY_SERP_API_KEY not set; cannot enable SerpAPI proxy")
            else:
                configured = pg.SerpAPI(api_key)
        elif mode == "tor":
            password = os.environ.get("SCHOLARLY_TOR_PASSWORD")
            socks_host = os.environ.get("SCHOLARLY_TOR_HOST", "127.0.0.1")
            socks_port = int(os.environ.get("SCHOLARLY_TOR_SOCKS_PORT", "9050"))
            configured = pg.Tor_Internal(password=password, socks_host=socks_host, socks_port=socks_port)
        elif mode == "tor-external":
            password = os.environ.get("SCHOLARLY_TOR_PASSWORD")
            socks_port = int(os.environ.get("SCHOLARLY_TOR_SOCKS_PORT", "9050"))
            control_port = int(os.environ.get("SCHOLARLY_TOR_CONTROL_PORT", "9051"))
            configured = pg.Tor_External(password=password, tor_sockport=socks_port, tor_control_port=control_port)
        elif mode == "http":
            proxy = os.environ.get("SCHOLARLY_HTTP_PROXY")
            if proxy:
                configured = pg.SingleProxy(http=proxy)
            else:
                logger.warning("SCHOLARLY_HTTP_PROXY not set; cannot enable HTTP proxy")
        else:
            logger.warning("Unknown SCHOLARLY_PROXY_MODE '%s'", mode)
            return
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to configure scholarly proxy mode %s: %s", mode, exc)
        configured = False

    if configured:
        scholarly.use_proxy(pg)
        logger.info("Configured scholarly proxy mode '%s'", mode)
    else:
        logger.warning("Could not configure scholarly proxy mode '%s'", mode)


_configure_scholarly_proxy()


@dataclass
class BibEntry:
    id: str
    title: Optional[str] = None
    author: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    raw_entry: str = ""


@dataclass
class CrossRefSource:
    title: Optional[str] = None
    journal: Optional[str] = None
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    doi: Optional[str] = None
    authors: Optional[List[str]] = field(default_factory=list)
    url: Optional[str] = None


@dataclass
class GoogleScholarSource:
    title: Optional[str] = None
    journal: Optional[str] = None
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    doi: Optional[str] = None
    authors: Optional[List[str]] = field(default_factory=list)
    url: Optional[str] = None
    citations: Optional[int] = None
    cited_by_url: Optional[str] = None


@dataclass
class ValidationResult:
    entry: BibEntry
    is_valid: bool
    validation_source: Optional[str] = None
    crossref_data: Optional[Dict[str, Any]] = None
    crossref_source: Optional[CrossRefSource] = None
    crossref_similarity_score: Optional[float] = None
    google_scholar_data: Optional[Dict[str, Any]] = None
    google_scholar_source: Optional[GoogleScholarSource] = None
    google_scholar_similarity_score: Optional[float] = None
    google_scholar_error: Optional[str] = None
    similarity_threshold: float = 75.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def _maybe_asdict(value: Any) -> Any:
            if value is None:
                return None
            if hasattr(value, "__dict__"):
                return {k: v for k, v in value.__dict__.items() if v is not None}
            return value

        return {
            "entry": _maybe_asdict(self.entry),
            "is_valid": self.is_valid,
            "validation_source": self.validation_source,
            "crossref_data": self.crossref_data,
            "crossref_source": _maybe_asdict(self.crossref_source),
            "crossref_similarity_score": self.crossref_similarity_score,
            "google_scholar_data": self.google_scholar_data,
            "google_scholar_source": _maybe_asdict(self.google_scholar_source),
            "google_scholar_similarity_score": self.google_scholar_similarity_score,
            "google_scholar_error": self.google_scholar_error,
            "similarity_threshold": self.similarity_threshold,
            "error_message": self.error_message,
        }


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\{|\}", "", text)
    text = re.sub(r"\\\\[a-zA-Z]+", "", text)
    text = re.sub(r"[^\w\s\u00C0-\u017F\u0100-\u024F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def extract_year_from_date(date_string: str) -> str:
    if not date_string:
        return ""
    match = re.search(r"(\d{4})", str(date_string))
    return match.group(1) if match else ""


async def query_crossref_by_doi(session: aiohttp.ClientSession, doi: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"https://api.crossref.org/works/{doi}"
        await _throttle_crossref_requests()
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("message", {})
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Error querying CrossRef by DOI %s: %s", doi, exc)
    return None


async def query_crossref_by_metadata(
    session: aiohttp.ClientSession, title: str, author: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    try:
        params: Dict[str, Any] = {"rows": 1}
        if title:
            params["query.title"] = title
        if author:
            params["query.author"] = author

        await _throttle_crossref_requests()
        async with session.get("https://api.crossref.org/works", params=params) as response:
            if response.status == 200:
                data = await response.json()
                items = data.get("message", {}).get("items", [])
                if items:
                    return items[0]
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Error querying CrossRef by metadata: %s", exc)
    return None


_scholar_last_request = 0.0
_scholar_lock = threading.Lock()


def rate_limit_scholar(min_interval: float = 2.0) -> None:
    global _scholar_last_request
    with _scholar_lock:
        now = time.time()
        elapsed = now - _scholar_last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _scholar_last_request = time.time()


def search_google_scholar(title: str, author: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if scholarly is None:
        return None, "scholarly package is not installed"

    query = title
    if author:
        query = f"{title} author:{author.split(',')[0].strip()}"

    try:
        rate_limit_scholar()
        logger.info("Searching Google Scholar for: %s", query)
        iterator = scholarly.search_pubs(query)
        try:
            result = next(iterator)
        except StopIteration:
            return None, f"no results for query '{query}'"

        if not result:
            return None, f"no results for query '{query}'"

        rate_limit_scholar()
        filled = scholarly.fill(result)
        return filled, None

    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Google Scholar search error: %s", exc)
        return None, f"Google Scholar error: {exc}"


def extract_google_scholar_source(data: Dict[str, Any]) -> GoogleScholarSource:
    try:
        bib = data.get("bib", {})
        title = bib.get("title")
        authors: List[str] = []
        author_field = bib.get("author")
        if isinstance(author_field, str):
            authors = [name.strip() for name in author_field.split(" and ") if name.strip()]
        elif isinstance(author_field, list):
            authors = [str(author).strip() for author in author_field]

        journal = bib.get("venue") or bib.get("journal")
        publisher = bib.get("publisher")
        published_date = bib.get("pub_year")
        doi = None
        if "eprint_url" in data and "doi.org" in str(data["eprint_url"]):
            doi = str(data["eprint_url"]).split("doi.org/")[-1]
        url = data.get("pub_url") or data.get("eprint_url")

        return GoogleScholarSource(
            title=title,
            journal=journal,
            publisher=publisher,
            published_date=str(published_date) if published_date else None,
            doi=doi,
            authors=authors,
            url=url,
            citations=data.get("num_citations"),
            cited_by_url=data.get("citedby_url"),
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Error extracting Google Scholar source: %s", exc)
        return GoogleScholarSource()


def calculate_similarity_score(entry: BibEntry, source: CrossRefSource) -> float:
    scores: List[float] = []
    weights: List[float] = []

    if entry.title and source.title:
        scores.append(fuzz.token_sort_ratio(normalize_text(entry.title), normalize_text(source.title)))
        weights.append(0.5)

    if entry.author and source.authors:
        normalized_author = normalize_text(entry.author)
        author_scores = [
            fuzz.token_sort_ratio(normalized_author, normalize_text(author))
            for author in source.authors
        ]
        if author_scores:
            scores.append(max(author_scores))
            weights.append(0.3)

    if entry.year and source.published_date:
        year = extract_year_from_date(source.published_date)
        if year:
            scores.append(100.0 if entry.year.strip() == year else 0.0)
            weights.append(0.15)

    if entry.journal and source.journal:
        scores.append(fuzz.token_sort_ratio(normalize_text(entry.journal), normalize_text(source.journal)))
        weights.append(0.05)

    if scores and weights:
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight else 0.0
    return 0.0


def calculate_google_scholar_similarity_score(entry: BibEntry, source: GoogleScholarSource) -> float:
    scores: List[float] = []
    weights: List[float] = []

    if entry.title and source.title:
        scores.append(fuzz.token_sort_ratio(normalize_text(entry.title), normalize_text(source.title)))
        weights.append(0.5)

    if entry.author and source.authors:
        normalized_author = normalize_text(entry.author)
        author_scores = [
            fuzz.token_sort_ratio(normalized_author, normalize_text(author))
            for author in source.authors
        ]
        if author_scores:
            scores.append(max(author_scores))
            weights.append(0.3)

    if entry.year and source.published_date:
        year = extract_year_from_date(source.published_date)
        if year:
            scores.append(100.0 if entry.year.strip() == year else 0.0)
            weights.append(0.15)

    if entry.journal and source.journal:
        scores.append(fuzz.token_sort_ratio(normalize_text(entry.journal), normalize_text(source.journal)))
        weights.append(0.05)

    if scores and weights:
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight else 0.0
    return 0.0


def extract_crossref_source(data: Dict[str, Any]) -> CrossRefSource:
    try:
        title = None
        if data.get("title"):
            title = data["title"][0] if isinstance(data["title"], list) else str(data["title"])

        journal = None
        if data.get("container-title"):
            journal = (
                data["container-title"][0]
                if isinstance(data["container-title"], list)
                else str(data["container-title"])
            )

        publisher = data.get("publisher")
        published_date = None
        for key in ("published-print", "published-online", "published"):
            if key in data:
                parts = data[key].get("date-parts", [])
                if parts and parts[0]:
                    published_date = "-".join(map(str, parts[0]))
                    break

        doi = data.get("DOI")
        authors: List[str] = []
        for author in data.get("author", []) or []:
            literal = author.get("literal")
            if literal:
                cleaned = str(literal).strip()
                if cleaned:
                    authors.append(cleaned)
                continue

            name = author.get("name")
            if name:
                cleaned = str(name).strip()
                if cleaned:
                    authors.append(cleaned)
                continue

            parts: List[str] = []
            given = author.get("given")
            family = author.get("family")
            if given:
                parts.append(str(given).strip())
            if family:
                parts.append(str(family).strip())

            if parts:
                cleaned = " ".join(part for part in parts if part)
                if cleaned:
                    authors.append(cleaned)

        authors = [name for name in dict.fromkeys(authors) if name]

        url = data.get("URL")

        return CrossRefSource(
            title=title,
            journal=journal,
            publisher=publisher,
            published_date=published_date,
            doi=doi,
            authors=authors,
            url=url,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Error extracting CrossRef source: %s", exc)
        return CrossRefSource()


async def validate_bib_entry(
    session: aiohttp.ClientSession,
    entry: BibEntry,
    similarity_threshold: float = 75.0,
    use_crossref: bool = True,
    use_scholar: bool = True,
) -> ValidationResult:
    crossref_data: Optional[Dict[str, Any]] = None
    crossref_source: Optional[CrossRefSource] = None
    crossref_similarity_score: Optional[float] = None
    google_scholar_data: Optional[Dict[str, Any]] = None
    google_scholar_source: Optional[GoogleScholarSource] = None
    google_scholar_similarity_score: Optional[float] = None
    google_scholar_error: Optional[str] = None
    is_valid = False
    validation_source: Optional[str] = None
    error_message: Optional[str] = None

    crossref_threshold_used: Optional[float] = None

    try:
        if use_crossref and entry.doi:
            clean_doi = entry.doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()
            crossref_data = await query_crossref_by_doi(session, clean_doi)
            if crossref_data:
                crossref_source = extract_crossref_source(crossref_data)
                crossref_similarity_score = calculate_similarity_score(entry, crossref_source)
                crossref_threshold_used = 60.0
                if crossref_similarity_score >= crossref_threshold_used:
                    is_valid = True
                    validation_source = "crossref"

        if use_crossref and not is_valid and entry.title:
            crossref_data = await query_crossref_by_metadata(session, entry.title, entry.author)
            if crossref_data:
                crossref_source = extract_crossref_source(crossref_data)
                crossref_similarity_score = calculate_similarity_score(entry, crossref_source)
                crossref_threshold_used = similarity_threshold
                if crossref_similarity_score >= crossref_threshold_used:
                    is_valid = True
                    validation_source = "crossref"

        if use_scholar and not is_valid and entry.title:
            loop = asyncio.get_event_loop()
            google_scholar_data, google_scholar_error = await loop.run_in_executor(
                None,
                search_google_scholar,
                entry.title,
                entry.author,
            )
            if google_scholar_data:
                google_scholar_source = extract_google_scholar_source(google_scholar_data)
                google_scholar_similarity_score = calculate_google_scholar_similarity_score(
                    entry, google_scholar_source
                )
                if google_scholar_similarity_score >= similarity_threshold:
                    is_valid = True
                    validation_source = "google_scholar"

        if not is_valid:
            messages: List[str] = []

            if use_crossref:
                if crossref_similarity_score is not None:
                    threshold_used = crossref_threshold_used or similarity_threshold
                    messages.append(
                        "CrossRef similarity "
                        f"{crossref_similarity_score:.1f}% < {threshold_used:.1f}%"
                    )
                else:
                    messages.append("No CrossRef match found")
            else:
                messages.append("CrossRef lookup disabled")

            if use_scholar:
                if entry.title:
                    if google_scholar_similarity_score is not None:
                        messages.append(
                            "Google Scholar similarity "
                            f"{google_scholar_similarity_score:.1f}% < {similarity_threshold:.1f}%"
                        )
                    else:
                        reason = google_scholar_error or "search returned no results"
                        messages.append(f"Google Scholar search failed ({reason})")
                else:
                    messages.append("Entry missing title; Google Scholar lookup skipped")
            else:
                messages.append("Google Scholar lookup disabled")

            error_message = "; ".join(messages) if messages else "Unable to validate entry"

    except Exception as exc:  # pragma: no cover - defensive guard
        error_message = f"Error validating entry: {exc}"
        logger.error("Error validating entry %s: %s", entry.id, exc)

    return ValidationResult(
        entry=entry,
        is_valid=is_valid,
        validation_source=validation_source,
        crossref_data=crossref_data,
        crossref_source=crossref_source,
        crossref_similarity_score=crossref_similarity_score,
        google_scholar_data=google_scholar_data,
        google_scholar_source=google_scholar_source,
        google_scholar_similarity_score=google_scholar_similarity_score,
        google_scholar_error=google_scholar_error,
        similarity_threshold=similarity_threshold,
        error_message=error_message,
    )


def _format_bibtex_entry(entry_dict: Dict[str, Any]) -> str:
    entry_type = str(entry_dict.get("ENTRYTYPE", "misc") or "misc")
    entry_id = str(entry_dict.get("ID", ""))

    body_lines: List[str] = []
    for key, value in entry_dict.items():
        if key in {"ENTRYTYPE", "ID"}:
            continue
        normalized_value = value
        if isinstance(normalized_value, (list, tuple)):
            normalized_value = " and ".join(str(item) for item in normalized_value)
        normalized_value = "" if normalized_value is None else str(normalized_value)
        body_lines.append(f"  {key} = {{{normalized_value}}},")

    if body_lines:
        body_lines[-1] = body_lines[-1].rstrip(",")

    lines = [f"@{entry_type}{{{entry_id},"]
    lines.extend(body_lines)
    lines.append("}")
    return "\n".join(lines)


def parse_bib_content(bib_content: str) -> List[BibEntry]:
    parser = BibTexParser()
    parser.ignore_nonstandard_types = False
    parser.homogenize_fields = True

    bib_database = bibtexparser.loads(bib_content, parser=parser)
    entries: List[BibEntry] = []

    for entry in bib_database.entries:
        raw_entry = _format_bibtex_entry(entry)
        bib_entry = BibEntry(
            id=entry.get("ID", str(uuid.uuid4())),
            title=entry.get("title", "").replace("{", "").replace("}", ""),
            author=entry.get("author", ""),
            journal=entry.get("journal", ""),
            year=entry.get("year", ""),
            doi=entry.get("doi", ""),
            url=entry.get("url", ""),
            raw_entry=raw_entry,
        )
        entries.append(bib_entry)

    return entries


async def validate_entries(
    entries: List[BibEntry],
    similarity_threshold: float = 75.0,
    use_crossref: bool = True,
    use_scholar: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    if not use_crossref and not use_scholar:
        raise ValueError("At least one of CrossRef or Google Scholar must be enabled")

    start = time.time()
    total_entries = len(entries)
    validation_results: List[ValidationResult] = []
    global _crossref_request_count, _crossref_rate_lock
    _crossref_request_count = 0
    _crossref_rate_lock = None
    ssl_context = _create_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else None
    async with aiohttp.ClientSession(connector=connector) as session:
        if total_entries > 0:
            tasks: List[asyncio.Task[ValidationResult]] = []
            completed = 0

            def _on_task_done(_: asyncio.Task[ValidationResult]) -> None:
                nonlocal completed
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_entries)

            for entry in entries:
                task = asyncio.create_task(
                    validate_bib_entry(
                        session,
                        entry,
                        similarity_threshold,
                        use_crossref=use_crossref,
                        use_scholar=use_scholar,
                    )
                )
                task.add_done_callback(_on_task_done)
                tasks.append(task)

            validation_results = await asyncio.gather(*tasks)

    valid_count = sum(1 for result in validation_results if result.is_valid)
    processing_time = time.time() - start

    return {
        "total_entries": len(entries),
        "valid_entries": valid_count,
        "invalid_entries": len(entries) - valid_count,
        "results": [result.to_dict() for result in validation_results],
        "processing_time": processing_time,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }