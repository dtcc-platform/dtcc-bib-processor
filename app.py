import streamlit as st
import asyncio
import threading
import json
import csv
import io
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import validation modules
try:
    from validator_core import parse_bib_content, validate_entries
except ImportError:
    st.error("""
    Missing required module: validator_core.py

    Please ensure validator_core.py is in the same directory as this app.
    """)
    st.stop()

st.set_page_config(
    page_title="DTCC Bib Processor",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'parsed_entries' not in st.session_state:
    st.session_state.parsed_entries = None
if 'filter_mode' not in st.session_state:
    st.session_state.filter_mode = "all"


def run_validation_in_thread(
    entries: List[Dict[str, Any]],
    threshold: float,
    use_crossref: bool,
    use_scholar: bool,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Run async validation in a separate thread to avoid event loop conflicts.
    """
    result = None
    exception = None

    def _run():
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                validate_entries(
                    entries,
                    threshold,
                    use_crossref=use_crossref,
                    use_scholar=use_scholar,
                    progress_callback=progress_callback
                )
            )
        except Exception as e:
            exception = e
        finally:
            loop.close()

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result


def format_authors(entry: Dict[str, Any]) -> str:
    """Format authors field for display."""
    author_field = entry.get("author")
    if not author_field:
        return "*No authors*"
    authors = [part.strip() for part in str(author_field).split(" and ") if part.strip()]
    if len(authors) > 2:
        return f"{authors[0]} et al."
    return ", ".join(authors) if authors else str(author_field)


def get_best_match(item: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], Optional[float]]]:
    """Get the best matching source for a validation item."""
    candidates = []

    crossref_source = item.get("crossref_source")
    if isinstance(crossref_source, dict) and crossref_source:
        candidates.append(("CrossRef", crossref_source, item.get("crossref_similarity_score")))

    scholar_source = item.get("google_scholar_source")
    if isinstance(scholar_source, dict) and scholar_source:
        candidates.append(("Google Scholar", scholar_source, item.get("google_scholar_similarity_score")))

    if not candidates:
        return None

    # Return the match with highest score
    return max(candidates, key=lambda x: x[2] if x[2] is not None else 0)


def export_to_csv(results: Dict[str, Any]) -> str:
    """Convert validation results to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "Title", "Authors", "Year", "Status", "Validation Source",
        "CrossRef Score", "Google Scholar Score", "DOI", "Publisher"
    ])

    # Write data rows
    for item in results.get("results", []):
        entry = item.get("entry", {})
        best_match = get_best_match(item)

        writer.writerow([
            entry.get("title", ""),
            format_authors(entry),
            entry.get("year", ""),
            "VALID" if item.get("is_valid") else "INVALID",
            item.get("validation_source", ""),
            f"{item.get('crossref_similarity_score', '')}",
            f"{item.get('google_scholar_similarity_score', '')}",
            entry.get("doi", ""),
            best_match[1].get("publisher", "") if best_match else ""
        ])

    return output.getvalue()


# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=60,
        max_value=95,
        value=75,
        step=5,
        help="Minimum similarity score to consider a match valid"
    )

    st.subheader("Validation Sources")
    use_crossref = st.checkbox("Use CrossRef", value=True)
    use_scholar = st.checkbox("Use Google Scholar", value=True)

    if not use_crossref and not use_scholar:
        st.error("At least one validation source must be selected")

    with st.expander("Advanced Options"):
        st.info("Proxy configuration for Google Scholar (if needed)")
        proxy_host = st.text_input("Proxy Host", placeholder="Optional")
        proxy_port = st.number_input("Proxy Port", min_value=0, max_value=65535, value=0)

    st.divider()
    st.caption("DTCC Bib Processor v1.0")
    st.caption("Validates bibliography entries against CrossRef and Google Scholar")


# Main content area
st.title("📚 DTCC Bib Processor")
st.write("Upload a .bib file to validate bibliography entries against academic databases.")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a .bib file",
    type=["bib"],
    help="Select a BibTeX file to validate"
)

if uploaded_file is not None:
    # Read and parse file content
    content = uploaded_file.read().decode("utf-8")

    # Parse entries if not already done
    if st.session_state.parsed_entries is None:
        with st.spinner("Parsing BibTeX file..."):
            entries = parse_bib_content(content)
            if not entries:
                st.error("No valid bibliography entries found in the file.")
                st.stop()
            st.session_state.parsed_entries = entries

    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("Total Entries", len(st.session_state.parsed_entries))
    with col3:
        st.metric("File Size", f"{len(content):,} bytes")

    # Preview section
    with st.expander("File Preview", expanded=False):
        st.text_area("Raw content (first 1000 characters)", content[:1000], height=200)

    # Validation section
    st.divider()

    if st.button("🔍 Validate Entries", type="primary", disabled=(not use_crossref and not use_scholar)):
        # Clear previous results
        st.session_state.validation_results = None
        st.session_state.filter_mode = "all"

        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Define progress callback
        def update_progress(completed: int, total: int):
            progress = completed / total if total > 0 else 1.0
            progress_bar.progress(progress)
            status_text.text(f"Validating: {completed}/{total} entries")

        try:
            # Run validation
            with st.spinner("Starting validation..."):
                results = run_validation_in_thread(
                    st.session_state.parsed_entries,
                    similarity_threshold,
                    use_crossref,
                    use_scholar,
                    update_progress
                )

            # Store results
            st.session_state.validation_results = results

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show success message
            st.success(f"✅ Validation complete! Processing time: {results.get('processing_time', 0):.2f}s")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Validation failed: {str(e)}")
            st.stop()

    # Results display section
    if st.session_state.validation_results:
        results = st.session_state.validation_results

        st.divider()
        st.header("Validation Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entries", results.get("total_entries", 0))
        with col2:
            valid_count = results.get("valid_entries", 0)
            st.metric("✅ Valid", valid_count, delta_color="normal")
        with col3:
            invalid_count = results.get("invalid_entries", 0)
            st.metric("❌ Invalid", invalid_count, delta_color="inverse")
        with col4:
            st.metric("⏱️ Time", f"{results.get('processing_time', 0):.1f}s")

        # Export buttons
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        with col2:
            csv_str = export_to_csv(results)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_str,
                file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Filter controls
        st.divider()
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            st.session_state.filter_mode = st.radio(
                "Show entries:",
                ["all", "valid", "invalid"],
                format_func=lambda x: {"all": "All", "valid": "✅ Valid Only", "invalid": "❌ Invalid Only"}[x]
            )

        # Entry list
        st.divider()
        entries_to_show = results.get("results", [])

        # Apply filter
        if st.session_state.filter_mode == "valid":
            entries_to_show = [e for e in entries_to_show if e.get("is_valid")]
        elif st.session_state.filter_mode == "invalid":
            entries_to_show = [e for e in entries_to_show if not e.get("is_valid")]

        st.write(f"Showing {len(entries_to_show)} entries:")

        # Display each entry
        for idx, item in enumerate(entries_to_show, 1):
            entry = item.get("entry", {})
            is_valid = item.get("is_valid")

            # Create expander with status icon
            status_icon = "✅" if is_valid else "❌"
            title = entry.get("title", "Untitled")
            if len(title) > 80:
                title = title[:77] + "..."

            with st.expander(f"{status_icon} [{idx}] {title}"):
                # Entry metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Authors:**", format_authors(entry))
                    st.write("**Year:**", entry.get("year", "N/A"))
                    if entry.get("doi"):
                        st.write("**DOI:**", entry.get("doi"))
                with col2:
                    st.write("**Status:**", "✅ VALID" if is_valid else "❌ INVALID")
                    if item.get("validation_source"):
                        st.write("**Validated by:**", item.get("validation_source"))

                # Similarity scores
                if item.get("crossref_similarity_score") is not None:
                    st.progress(item["crossref_similarity_score"]/100, text=f"CrossRef similarity: {item['crossref_similarity_score']:.1f}%")
                if item.get("google_scholar_similarity_score") is not None:
                    st.progress(item["google_scholar_similarity_score"]/100, text=f"Google Scholar similarity: {item['google_scholar_similarity_score']:.1f}%")

                # Best match details
                best_match = get_best_match(item)
                if best_match:
                    source_name, match_data, score = best_match
                    st.write(f"**Best match from {source_name}:**")

                    match_col1, match_col2 = st.columns(2)
                    with match_col1:
                        if match_data.get("title"):
                            st.write("Title:", match_data["title"])
                        if match_data.get("journal"):
                            st.write("Journal:", match_data["journal"])
                        if match_data.get("publisher"):
                            st.write("Publisher:", match_data["publisher"])
                    with match_col2:
                        if match_data.get("published_date"):
                            st.write("Published:", match_data["published_date"])
                        if match_data.get("doi"):
                            st.write("DOI:", match_data["doi"])
                        if match_data.get("citations") is not None:
                            st.write("Citations:", match_data["citations"])

                # Error message if any
                if item.get("error_message"):
                    st.error(f"Error: {item['error_message']}")

                # Google Scholar error if any
                if item.get("google_scholar_error"):
                    st.warning(f"Google Scholar: {item['google_scholar_error']}")

                # Raw BibTeX entry
                if entry.get("raw_entry"):
                    st.code(entry["raw_entry"], language="bibtex")

else:
    # No file uploaded - show instructions
    st.info("👆 Please upload a .bib file to begin validation")

    with st.expander("ℹ️ How it works"):
        st.write("""
        This tool validates bibliography entries by:

        1. **Parsing** your .bib file to extract individual entries
        2. **Searching** CrossRef and/or Google Scholar for matching publications
        3. **Comparing** titles, authors, and metadata using fuzzy matching
        4. **Reporting** validation status with similarity scores

        **Tips:**
        - Adjust the similarity threshold for stricter or more lenient matching
        - Use both CrossRef and Google Scholar for best coverage
        - Export results as JSON for detailed analysis or CSV for spreadsheet review
        """)
