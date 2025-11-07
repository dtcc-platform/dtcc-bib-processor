# DTCC Bib Processor

A Streamlit web application for validating BibTeX bibliography entries against the CrossRef database.

## Features

- Upload and parse `.bib` files
- Validate entries against CrossRef database
- Fuzzy matching with adjustable similarity threshold
- Export results as JSON or CSV
- Interactive results display with expandable details

## Installation

1. Install Python 3.7+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How It Works

1. **Upload**: Select a .bib file using the file uploader
2. **Configure**: Adjust similarity threshold (60-95%) in the sidebar
3. **Validate**: Click "Validate Entries" to process
4. **Review**: Browse results with filter options
5. **Export**: Download validation results as JSON or CSV

## Files

- `app.py` - Main Streamlit application
- `validator_core.py` - Core validation logic
- `validate.py` - Command-line validation tool
- `sample.bib` - Sample bibliography for testing
- `requirements.txt` - Python dependencies

## Requirements

- streamlit - Web application framework
- aiohttp - Async HTTP client for CrossRef API
- bibtexparser - BibTeX file parsing
- rapidfuzz - Fuzzy string matching
- certifi - SSL certificates
- packaging - Version handling
- tqdm - Progress bars

## Note

This application uses CrossRef for validation, which provides excellent coverage for academic papers with DOIs. Papers without DOIs or those not indexed in CrossRef may not be found.