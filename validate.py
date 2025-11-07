#!/usr/bin/env python3
"""Validate bibliography references locally using CrossRef."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate bibliography entries directly against CrossRef",
    )
    parser.add_argument(
        "bibfile",
        type=Path,
        help="Path to the .bib file to validate",
    )
    parser.add_argument(
        "--similarity-threshold",
        dest="threshold",
        type=float,
        default=75.0,
        help="Similarity threshold for fuzzy matching (default: 75.0)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON response instead of the human-readable summary",
    )
    return parser.parse_args()


def validate_bib_file(
    bib_path: Path,
    threshold: float,
) -> Dict[str, Any]:
    from validator_core import parse_bib_content, validate_entries  # Local import for friendlier CLI errors

    if not bib_path.exists():
        raise FileNotFoundError(f"No such file: {bib_path}")

    if not bib_path.is_file():
        raise ValueError(f"Path is not a file: {bib_path}")

    if bib_path.suffix.lower() != ".bib":
        raise ValueError("Input file must have a .bib extension")

    content = bib_path.read_text(encoding="utf-8")
    entries = parse_bib_content(content)
    if not entries:
        raise ValueError("No valid bibliography entries found in file")

    progress_callback: Optional[Callable[[int, int], None]] = None
    progress_bar = None
    simple_progress_used = False

    try:
        from tqdm import tqdm as tqdm_cls  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        tqdm_cls = None  # type: ignore

    if entries:
        if tqdm_cls is not None:
            progress_bar = tqdm_cls(total=len(entries), desc="Validating", unit="entry")
            last_completed = 0

            def _update_progress(completed: int, total: int) -> None:
                nonlocal last_completed
                if progress_bar is not None:
                    progress_bar.update(completed - last_completed)
                last_completed = completed

            progress_callback = _update_progress
        else:
            simple_progress_used = True

            def _update_progress(completed: int, total: int) -> None:
                percent = (completed / total) * 100 if total else 100
                print(
                    f"\rValidating entries: {completed}/{total} ({percent:.0f}%)",
                    end="",
                    flush=True,
                )

            progress_callback = _update_progress

    try:
        return asyncio.run(
            validate_entries(
                entries,
                threshold,
                use_crossref=True,
                progress_callback=progress_callback,
            )
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if simple_progress_used:
            print(flush=True)


def _format_authors(entry: Dict[str, Any]) -> str:
    author_field = entry.get("author")
    if not author_field:
        return "<missing>"
    authors = [part.strip() for part in str(author_field).split(" and ") if part.strip()]
    return ", ".join(authors) if authors else str(author_field)


def _select_best_match(item: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], Optional[float]]]:
    crossref_source = item.get("crossref_source")
    if isinstance(crossref_source, dict) and crossref_source:
        return ("CrossRef", crossref_source, item.get("crossref_similarity_score"))
    return None


def _format_match_details(match: Dict[str, Any]) -> List[str]:
    details: List[str] = []

    title = match.get("title")
    if title:
        details.append(f"Title: {title}")

    authors = match.get("authors")
    if isinstance(authors, list) and authors:
        details.append(f"Authors: {', '.join(str(author) for author in authors)}")

    journal = match.get("journal")
    if journal:
        details.append(f"Journal: {journal}")

    publisher = match.get("publisher")
    if publisher:
        details.append(f"Publisher: {publisher}")

    published_date = match.get("published_date")
    if published_date:
        details.append(f"Published: {published_date}")

    doi = match.get("doi")
    if doi:
        details.append(f"DOI: {doi}")

    url = match.get("url")
    if url:
        details.append(f"URL: {url}")

    if match.get("citations") is not None:
        details.append(f"Citations: {match['citations']}")

    cited_by_url = match.get("cited_by_url")
    if cited_by_url:
        details.append(f"Cited by URL: {cited_by_url}")

    return details or ["<no metadata>"]


def print_summary(result: Dict[str, Any]) -> None:
    print(f"Total entries: {result.get('total_entries')}")
    print(f"Valid entries: {result.get('valid_entries')}")
    print(f"Invalid entries: {result.get('invalid_entries')}")
    print(f"Processing time: {result.get('processing_time'):.2f}s")
    generated_at = result.get("generated_at")
    if generated_at:
        print(f"Generated at: {generated_at}")
    print("Source: CrossRef")
    print()

    results: List[Dict[str, Any]] = result.get("results", [])
    invalid_summaries: List[Tuple[int, str, Optional[str], Optional[float], Optional[str]]] = []
    for index, item in enumerate(results, start=1):
        entry = item.get("entry", {})
        title = entry.get("title") or "<no title>"
        status = "VALID" if item.get("is_valid") else "INVALID"
        print(f"[{index}] {title}")
        print(f"    Status: {status}")
        print(f"    Authors: {_format_authors(entry)}")
        if entry.get("year"):
            print(f"    Year: {entry['year']}")
        if entry.get("doi"):
            print(f"    DOI: {entry['doi']}")
        validation_source = item.get("validation_source")
        if validation_source:
            print(f"    Validation source: {validation_source}")
        if item.get("crossref_similarity_score") is not None:
            print(f"    CrossRef similarity: {item['crossref_similarity_score']:.1f}%")
        best_match = _select_best_match(item)
        if best_match:
            source_name, match_data, match_score = best_match
            header = f"    Closest match ({source_name}"
            if isinstance(match_score, (int, float)):
                header += f", {match_score:.1f}%"
            header += ")"
            print(header)
            for detail in _format_match_details(match_data):
                print(f"        {detail}")
        if not item.get("is_valid"):
            best_source = None
            best_score: Optional[float] = None
            if best_match:
                best_source = best_match[0]
                if isinstance(best_match[2], (int, float)):
                    best_score = float(best_match[2])
            raw_entry = entry.get("raw_entry") if isinstance(entry, dict) else None
            raw_entry_str = str(raw_entry) if raw_entry not in (None, "") else None
            invalid_summaries.append((index, title, best_source, best_score, raw_entry_str))
        error_message = item.get("error_message")
        if error_message:
            print(f"    Error: {error_message}")
        print()

    print("Summary")
    print(f"Valid references: {result.get('valid_entries', 0)}")
    print(f"Invalid references: {result.get('invalid_entries', 0)}")
    if invalid_summaries:
        print("Invalid reference matches:")
        for idx, title, source, score, raw_entry in invalid_summaries:
            label = source if source else "No match"
            if isinstance(score, (int, float)):
                print(f"  - [{idx}] {title} — {label}: {score:.1f}%")
            else:
                print(f"  - [{idx}] {title} — {label}: <no score>")
            if raw_entry:
                print("    Bib entry:")
                for line in raw_entry.splitlines():
                    print(f"      {line}")


def main() -> None:
    args = parse_args()

    try:
        results = validate_bib_file(args.bibfile, args.threshold)
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error: {exc}")
        raise SystemExit(1)

    if args.raw:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()