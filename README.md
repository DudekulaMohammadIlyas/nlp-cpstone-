# Explainable Multilingual NLP Intelligence (T5 Summarizer Project)

## Project Overview

This repository contains a Flask-based web application that performs explainable NLP summarization, sentence importance ranking, part-of-speech analysis, parse-tree visualization, sentiment detection, translation, semantic similarity checks, and downloadable PDF report generation. The app accepts raw text or uploaded `.txt`, `.pdf`, and `.docx` files and provides interactive, explainable outputs for each processing stage.

## Key Features

- Upload or paste text and get an extractive summary.
- Sentence importance ranking using TF-IDF.
- Part-of-speech (POS) grouping and visualizations.
- Parse-tree building and rendering (diagram output included in PDF).
- Sentiment (tone) estimation.
- Machine translation with language selection and translation similarity score.
- Named-entity-like extraction (proper-noun detection) for English.
- Download a comprehensive PDF report including charts and diagrams.

## Technologies & Libraries

- Web framework: `Flask` (routing and templating with Jinja2)
- NLP: `nltk` (tokenization, POS tagging, stopwords, parsing helpers)
- Text processing / translation: `TextBlob`, `deep_translator` (GoogleTranslator)
- File parsing: `docx2txt` (DOCX), `PyPDF2` (PDF)
- ML / similarity: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
- Visualization: `matplotlib` (charts and parse-tree diagrams)
- PDF generation: `reportlab` (PDF report, custom fonts support)
- Utilities: `io`, `tempfile`, `json`, `collections.Counter`

## Project Structure

- `app.py` — Main Flask application and all route handlers.
- `templates/` — Jinja2 HTML templates used by the routes (includes `module0.html`, `module1.html`, `module2.html`, `module3.html`, `dashboard.html`, `index.html`, `base.html`).
- `static/` — Static assets (CSS, images) used by templates; contains `style.css`.
- `fonts/` — Fonts directory (project uses `NotoSans-Regular.ttf` by default for PDF unicode rendering).
- `README.md` — (this file) project documentation.

## Module / Route Breakdown (Technologies Used)

- `GET /` — Home (template: `module0.html`)
  - Purpose: Accepts pasted text or file upload.
  - Technologies: Flask form handling, `extract_text()` uses `docx2txt` and `PyPDF2`.

- `POST /` (from `/`) — Text ingestion and processing
  - Purpose: Runs sentence segmentation and scoring, builds summary, POS analysis, and parse-tree for the top sentence.
  - Technologies: `nltk` (`sent_tokenize`, `word_tokenize`, `pos_tag`), `scikit-learn` (`TfidfVectorizer`) for sentence importance, custom parse using `nltk.RegexpParser`.

- `GET /analysis` — Analysis view (template: `module1.html`)
  - Purpose: Shows sentence cards (importance, keywords), sentiment/tone, and parse-tree JSON for visualization.
  - Technologies: `TextBlob` for sentiment polarity, front-end rendering via Jinja2 and client-side JS (if present) for parse-tree display.

- `GET, POST /translation` — Translation UI (template: `module2.html`)
  - Purpose: Translate the generated summary to a selected language, compute semantic similarity between original summary and translation, and extract proper-noun like entities.
  - Technologies: `deep_translator.GoogleTranslator`, `scikit-learn` TF-IDF + cosine similarity for semantic similarity, simple NER via POS filtering.

- `GET /dashboard` — Dashboard (template: `dashboard.html`)
  - Purpose: Displays POS tag distribution and lists of important words per POS.
  - Technologies: `matplotlib` for generating graphs (images embedded in the PDF report), Jinja2 rendering for charts/values.

- `GET /download_report` — PDF report generation and download
  - Purpose: Compile a downloadable PDF containing the original text, generated summary, POS graph, parse-tree diagram, and POS-tagged words.
  - Technologies: `reportlab` to compose the PDF, `matplotlib` to render charts/diagrams, system font loading via `reportlab.pdfbase.ttfonts.TTFont`.

Note: The templates `module3.html` and `index.html` can be used for expansion or additional features; currently the primary routes map to the pages above.

## Installation & Setup (Quick)

1. Create a Python virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install required packages:

```bash
pip install flask nltk docx2txt PyPDF2 deep-translator textblob scikit-learn matplotlib reportlab
```

3. (Optional) If `textblob` errors on first use, run:

```bash
python -m textblob.download_corpora
```

4. Run the app:

```bash
python app.py
```

5. Open a browser at `http://127.0.0.1:5000/` and interact with the UI.

Notes:
- `app.py` already triggers `nltk.download()` calls for `punkt`, `averaged_perceptron_tagger`, and `stopwords` at runtime. If you prefer to pre-download corpora, use `nltk.download()` from a Python REPL.
- Ensure `fonts/NotoSans-Regular.ttf` exists for correct unicode rendering in PDFs. Replace the font in `app.py` if you prefer a different TrueType font.

## Usage Examples

- Paste or upload a document on the homepage to generate a summary and analysis.
- Use the translation page to translate the summary and inspect semantic similarity.
- Click the dashboard to view POS distributions and download the full PDF report via the provided button.

## Development & Contribution

- To add a new NLP module, create a new route in `app.py`, add a corresponding template in `templates/`, and wire any required static assets in `static/`.
- Keep heavy processing asynchronous if you integrate larger models or long-running tasks (e.g., Celery or background threads) to avoid blocking Flask's development server.

## Known Dependencies

- Python 3.8+
- Flask
- NLTK corpora: `punkt`, `averaged_perceptron_tagger`, `stopwords`

## License & Contact

This repository does not currently include a license file. Add `LICENSE` if you want an open-source license applied.

For questions or contributions, open an issue or submit a pull request.

---

If you'd like, I can also:
- Add a `requirements.txt` with pinned versions.
- Add a basic `README` section with deployment tips (Docker / Heroku / Azure).
- Extract the Flask app into a package with unit tests for core functions (`analyze_sentences`, `summarize_from_sentences`, etc.).
