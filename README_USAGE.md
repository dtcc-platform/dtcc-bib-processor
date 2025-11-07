# DTCC Bib Processor - Usage Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## Features

### Configuration (Sidebar)
- **Similarity Threshold**: Adjust matching strictness (60-95%)
- **Validation Source**: Uses CrossRef for reliable validation

### Main Interface

1. **Upload**: Select a .bib file using the file uploader
2. **Validate**: Click "Validate Entries" to process the bibliography
3. **View Results**:
   - Summary metrics (total, valid, invalid entries)
   - Filter entries (all/valid/invalid)
   - Expandable details for each entry
   - Download results as JSON or CSV

### Results Display

Each validated entry shows:
- Validation status (valid/invalid)
- Authors, year, DOI
- CrossRef similarity score
- Match details from CrossRef
- Raw BibTeX entry

## Sample File

A sample.bib file is included for testing with 5 entries:
- 4 real publications (should validate)
- 1 fake publication (should fail validation)

## Troubleshooting

- **Missing dependencies**: Run `pip install -r requirements.txt`
- **File parsing errors**: Ensure .bib file is properly formatted BibTeX
- **Validation failures**: Papers without DOIs may not be found in CrossRef