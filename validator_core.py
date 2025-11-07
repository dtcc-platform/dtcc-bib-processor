"""Simplified validator core using only CrossRef."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import bibtexparser
from bibtexparser.bparser import BibTexParser
from rapidfuzz import fuzz

try:
    import certifi
except ImportError:
    certifi = None

logger = logging.getLogger(__name__)

# Rate limiting for CrossRef
_crossref_request_count = 0
_crossref_rate_lock: Optional[asyncio.Lock] = None


@dataclass
class BibEntry:
    """Represents a bibliography entry."""
    title: str
    author: Optional[str]
    year: Optional[str]
    doi: Optional[str]
    journal: Optional[str]
    raw_entry: Dict[str, Any]


@dataclass
class CrossRefSource:
    """CrossRef search result."""
    title: str
    authors: List[str]
    journal: Optional[str]
    publisher: Optional[str]
    published_date: Optional[str]
    doi: Optional[str]
    url: Optional[str]


@dataclass
class ValidationResult:
    """Result of validating a single entry."""
    entry: BibEntry
    is_valid: bool
    validation_source: Optional[str]
    error_message: Optional[str]

    crossref_data: Optional[Dict[str, Any]] = None
    crossref_source: Optional[CrossRefSource] = None
    crossref_similarity_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": {
                "title": self.entry.title,
                "author": self.entry.author,
                "year": self.entry.year,
                "doi": self.entry.doi,
                "journal": self.entry.journal,
                "raw_entry": self.entry.raw_entry.get("ID", ""),
            },
            "is_valid": self.is_valid,
            "validation_source": self.validation_source,
            "error_message": self.error_message,
            "crossref_data": self.crossref_data,
            "crossref_source": self.crossref_source.__dict__ if self.crossref_source else None,
            "crossref_similarity_score": self.crossref_similarity_score,
        }


def parse_bib_content(content: str) -> List[BibEntry]:
    """Parse BibTeX content into entries."""
    parser = BibTexParser()
    bib_db = bibtexparser.loads(content, parser=parser)

    entries = []
    for entry in bib_db.entries:
        entries.append(
            BibEntry(
                title=entry.get("title", "").strip(),
                author=entry.get("author"),
                year=entry.get("year"),
                doi=entry.get("doi"),
                journal=entry.get("journal"),
                raw_entry=entry
            )
        )
    return entries


async def search_crossref(
    session: aiohttp.ClientSession,
    title: str,
    author: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Search CrossRef for a publication."""
    global _crossref_request_count, _crossref_rate_lock

    if _crossref_rate_lock is None:
        _crossref_rate_lock = asyncio.Lock()

    # Rate limiting
    async with _crossref_rate_lock:
        _crossref_request_count += 1
        if _crossref_request_count > 1:
            await asyncio.sleep(0.1)

    query = title
    if author:
        author_parts = author.split(" and ")
        if author_parts:
            query = f"{title} {author_parts[0].strip()}"

    params = {
        "query": query,
        "rows": 5,
        "select": "DOI,title,author,published-print,published-online,publisher,container-title,type"
    }

    url = "https://api.crossref.org/works"

    try:
        ssl_context = None
        if certifi:
            import ssl
            ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with session.get(url, params=params, ssl=ssl_context) as response:
            if response.status == 200:
                data = await response.json()
                items = data.get("message", {}).get("items", [])
                if items:
                    return items[0]
    except Exception as e:
        logger.error(f"CrossRef search error: {e}")

    return None


def extract_crossref_source(data: Dict[str, Any]) -> CrossRefSource:
    """Extract structured data from CrossRef response."""
    # Extract authors
    authors = []
    for author in data.get("author", []):
        name_parts = []
        if "given" in author:
            name_parts.append(author["given"])
        if "family" in author:
            name_parts.append(author["family"])
        if name_parts:
            authors.append(" ".join(name_parts))

    # Extract date
    date_parts = None
    for date_field in ["published-print", "published-online"]:
        if date_field in data and "date-parts" in data[date_field]:
            date_parts = data[date_field]["date-parts"]
            break

    published_date = None
    if date_parts and date_parts[0]:
        year = date_parts[0][0] if len(date_parts[0]) > 0 else None
        month = date_parts[0][1] if len(date_parts[0]) > 1 else None
        day = date_parts[0][2] if len(date_parts[0]) > 2 else None
        if year:
            published_date = str(year)
            if month:
                published_date += f"-{month:02d}"
                if day:
                    published_date += f"-{day:02d}"

    # Extract title
    titles = data.get("title", [])
    title = titles[0] if titles else ""

    return CrossRefSource(
        title=title,
        authors=authors,
        journal=data.get("container-title", [None])[0] if data.get("container-title") else None,
        publisher=data.get("publisher"),
        published_date=published_date,
        doi=data.get("DOI"),
        url=f"https://doi.org/{data.get('DOI')}" if data.get("DOI") else None
    )


def calculate_crossref_similarity_score(entry: BibEntry, source: CrossRefSource) -> float:
    """Calculate similarity score between entry and CrossRef result."""
    scores = []

    # Title similarity (highest weight)
    if entry.title and source.title:
        title_score = fuzz.ratio(
            entry.title.lower().strip(),
            source.title.lower().strip()
        )
        scores.append(title_score * 0.6)

    # Author similarity
    if entry.author and source.authors:
        author_score = 0
        entry_authors = [a.strip() for a in entry.author.split(" and ")]

        for entry_author in entry_authors:
            for source_author in source.authors:
                score = fuzz.partial_ratio(
                    entry_author.lower(),
                    source_author.lower()
                )
                author_score = max(author_score, score)

        scores.append(author_score * 0.3)

    # Year similarity
    if entry.year and source.published_date:
        source_year = source.published_date[:4]
        if entry.year == source_year:
            scores.append(100 * 0.1)
        elif abs(int(entry.year) - int(source_year)) <= 1:
            scores.append(50 * 0.1)

    return sum(scores) if scores else 0.0


async def validate_single_entry(
    session: aiohttp.ClientSession,
    entry: BibEntry,
    similarity_threshold: float
) -> ValidationResult:
    """Validate a single bibliography entry."""

    # Search CrossRef
    crossref_data = await search_crossref(session, entry.title, entry.author)

    if crossref_data:
        crossref_source = extract_crossref_source(crossref_data)
        crossref_similarity_score = calculate_crossref_similarity_score(entry, crossref_source)

        if crossref_similarity_score >= similarity_threshold:
            return ValidationResult(
                entry=entry,
                is_valid=True,
                validation_source="crossref",
                error_message=None,
                crossref_data=crossref_data,
                crossref_source=crossref_source,
                crossref_similarity_score=crossref_similarity_score
            )
        else:
            return ValidationResult(
                entry=entry,
                is_valid=False,
                validation_source=None,
                error_message=f"CrossRef similarity score {crossref_similarity_score:.1f}% < {similarity_threshold:.1f}%",
                crossref_data=crossref_data,
                crossref_source=crossref_source,
                crossref_similarity_score=crossref_similarity_score
            )
    else:
        return ValidationResult(
            entry=entry,
            is_valid=False,
            validation_source=None,
            error_message="No results found in CrossRef",
            crossref_data=None,
            crossref_source=None,
            crossref_similarity_score=None
        )


async def validate_entries(
    entries: List[BibEntry],
    similarity_threshold: float = 75.0,
    use_crossref: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """Validate multiple bibliography entries."""

    if not entries:
        return {
            "total_entries": 0,
            "valid_entries": 0,
            "invalid_entries": 0,
            "results": [],
            "processing_time": 0.0,
            "generated_at": datetime.utcnow().isoformat()
        }

    start_time = time.time()

    # Create session with timeout
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for i, entry in enumerate(entries):
            task = asyncio.create_task(
                validate_single_entry(session, entry, similarity_threshold)
            )
            tasks.append(task)

            # Progress callback
            if progress_callback:
                def _on_task_done(fut):
                    completed = sum(1 for t in tasks if t.done())
                    try:
                        progress_callback(completed, len(entries))
                    except Exception:
                        pass  # Ignore callback errors

                task.add_done_callback(_on_task_done)

        results = await asyncio.gather(*tasks)

    # Count valid/invalid
    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = len(results) - valid_count

    return {
        "total_entries": len(entries),
        "valid_entries": valid_count,
        "invalid_entries": invalid_count,
        "results": [r.to_dict() for r in results],
        "processing_time": time.time() - start_time,
        "generated_at": datetime.utcnow().isoformat()
    }