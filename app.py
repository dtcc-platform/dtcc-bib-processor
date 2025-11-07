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
    progress_placeholder = None,
    status_placeholder = None
) -> Dict[str, Any]:
    """
    Run async validation in a separate thread to avoid event loop conflicts.
    """
    result = None
    exception = None
    progress_queue = []

    def thread_safe_progress(completed: int, total: int):
        """Queue progress updates to be handled in main thread."""
        progress_queue.append((completed, total))

    def _run():
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                validate_entries(
                    entries,
                    threshold,
                    use_crossref=True,
                    progress_callback=thread_safe_progress if progress_placeholder else None
                )
            )
        except Exception as e:
            exception = e
        finally:
            loop.close()

    thread = threading.Thread(target=_run)
    thread.start()

    # Update progress in main thread
    if progress_placeholder and status_placeholder:
        while thread.is_alive():
            if progress_queue:
                completed, total = progress_queue.pop(0)
                progress = completed / total if total > 0 else 1.0
                progress_placeholder.progress(progress)
                status_placeholder.text(f"Validating: {completed}/{total} entries")
            thread.join(timeout=0.1)

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
    """Get the CrossRef match for a validation item."""
    crossref_source = item.get("crossref_source")
    if isinstance(crossref_source, dict) and crossref_source:
        return ("CrossRef", crossref_source, item.get("crossref_similarity_score"))
    return None


def export_to_csv(results: Dict[str, Any]) -> str:
    """Convert validation results to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "Title", "Authors", "Year", "Status",
        "CrossRef Score", "DOI", "Publisher"
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
            f"{item.get('crossref_similarity_score', '')}",
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

    st.subheader("Validation Source")
    st.info("✅ Using CrossRef for reliable bibliography validation")

    st.divider()
    st.caption("DTCC Bib Processor v1.0")
    st.caption("Validates bibliography entries against CrossRef")


# Main content area
st.title("📚 DTCC Bib Processor")
st.write("Upload a .bib file to validate bibliography entries against CrossRef database.")

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

    if st.button("🔍 Validate Entries", type="primary"):
        # Clear previous results
        st.session_state.validation_results = None
        st.session_state.filter_mode = "all"

        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Run validation with thread-safe progress updates
            with st.spinner("Starting validation..."):
                results = run_validation_in_thread(
                    st.session_state.parsed_entries,
                    similarity_threshold,
                    progress_bar,
                    status_text
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

        # Summary sections for valid and invalid entries
        st.divider()

        # Create tabs for valid and invalid summaries
        tab1, tab2 = st.tabs(["✅ Valid Entries", "❌ Invalid Entries"])

        with tab1:
            valid_entries = [e for e in results.get("results", []) if e.get("is_valid")]
            if valid_entries:
                st.write(f"Found **{len(valid_entries)}** valid entries with high-confidence CrossRef matches.")
                for item in valid_entries:
                    entry = item.get("entry", {})
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.success(f"**{entry.get('title', 'Untitled')[:100]}**")
                            st.caption(f"Authors: {format_authors(entry)} | Year: {entry.get('year', 'N/A')}")
                        with col2:
                            if item.get("crossref_similarity_score") is not None:
                                st.metric("Match", f"{item['crossref_similarity_score']:.1f}%")
            else:
                st.info("No valid entries found.")

        with tab2:
            invalid_entries = [e for e in results.get("results", []) if not e.get("is_valid")]
            if invalid_entries:
                st.write(f"Found **{len(invalid_entries)}** entries below the similarity threshold.")

                for item in invalid_entries:
                    entry = item.get("entry", {})
                    best_match = get_best_match(item)

                    with st.container():
                        # Entry title and score
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.error(f"**{entry.get('title', 'Untitled')[:100]}**")
                            st.caption(f"Authors: {format_authors(entry)}")
                        with col2:
                            if item.get("crossref_similarity_score") is not None:
                                st.metric("Score", f"{item['crossref_similarity_score']:.1f}%")

                        # Best match from CrossRef
                        if best_match:
                            source_name, match_data, score = best_match
                            st.info(f"**Closest CrossRef Match:**  \n"
                                   f"*{match_data.get('title', 'No title')}*  \n"
                                   f"Authors: {', '.join(match_data.get('authors', [])) if match_data.get('authors') else 'N/A'}  \n"
                                   f"Journal: {match_data.get('journal', 'N/A')} | Year: {match_data.get('published_date', 'N/A')[:4] if match_data.get('published_date') else 'N/A'}  \n"
                                   f"DOI: {match_data.get('doi', 'N/A')}")
                        else:
                            st.warning("No CrossRef match found")

                        st.divider()
            else:
                st.info("All entries were successfully validated.")

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
        st.subheader("Detailed Results")
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
            best_match = get_best_match(item)

            # Create expander with status icon
            status_icon = "✅" if is_valid else "❌"
            title = entry.get("title", "Untitled")
            if len(title) > 80:
                title = title[:77] + "..."

            # Add similarity score to the expander title if available
            expander_title = f"{status_icon} [{idx}] {title}"
            if item.get("crossref_similarity_score") is not None:
                expander_title += f" (Score: {item['crossref_similarity_score']:.1f}%)"

            with st.expander(expander_title):
                # Entry metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Authors:**", format_authors(entry))
                    st.write("**Year:**", entry.get("year", "N/A"))
                    if entry.get("doi"):
                        st.write("**DOI:**", entry.get("doi"))
                with col2:
                    st.write("**Status:**", "✅ VALID" if is_valid else "❌ INVALID")
                    st.write("**Source:**", "CrossRef")

                # Similarity score
                if item.get("crossref_similarity_score") is not None:
                    st.progress(item["crossref_similarity_score"]/100, text=f"CrossRef similarity: {item['crossref_similarity_score']:.1f}%")

                # Best match details
                best_match = get_best_match(item)
                if best_match:
                    source_name, match_data, score = best_match
                    st.write("**CrossRef Match Details:**")

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
        2. **Searching** CrossRef database for matching publications
        3. **Comparing** titles, authors, and metadata using fuzzy matching
        4. **Reporting** validation status with similarity scores

        **Tips:**
        - Adjust the similarity threshold for stricter or more lenient matching
        - CrossRef provides excellent coverage for academic papers with DOIs
        - Export results as JSON for detailed analysis or CSV for spreadsheet review
        """)
