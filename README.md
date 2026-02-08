# T5 Neural Summarizer & Multilingual Hub

An advanced natural language processing application combining **abstractive text summarization**, **linguistic analytics**, and **neural machine translation** into a unified Explainable AI pipeline.

**GitHub Repository**: [nlp-cpstone-](https://github.com/BandlapalliBhanutejareddy/nlp-cpstone-)

---

## Overview

This project leverages state-of-the-art transformer models to provide a complete text processing workflow:

- **Module 0**: Data Acquisition & Input Handling
- **Module 1**: Cleaning & Linguistic Analytics  
- **Module 2**: Abstractive Summarization using T5
- **Module 3**: Neural Multilingual Translation

The application features a responsive web interface for interactive text analysis with real-time performance metrics.

---

## Features

### Text Analysis & Cleaning (Module 1)
- **Deep Text Cleaning**: Removes special characters and normalizes whitespace
- **POS (Part-of-Speech) Tagging**: Visualizes grammatical structure with spaCy
- **Linguistic Statistics**:
  - Original word count
  - Cleaned word count
  - Sentence count
  - Noise reduction percentage
- **Real-time Processing Time**: Tracks execution speed for performance monitoring

### Abstractive Summarization (Module 2)
- **T5 Transformer Model**: Uses `t5-small` (60M parameters) for efficient, high-quality text summarization
- **Configurable Summary Length**: Token-based output (max 400 tokens, min 120 tokens)
- **Advanced Decoding Strategies**:
  - **Beam Search**: 3 beams for optimal speed-quality balance
  - **N-gram Blocking**: `no_repeat_ngram_size=3` prevents repetitive outputs
  - **Length Penalty**: 2.0 to encourage balanced summary length
  - **Early Stopping**: Terminates generation when sequences become redundant
- **Hallucination Prevention**: N-gram blocking and length constraints ensure factually grounded output

### Multilingual Translation (Module 3)
- **Supported Languages**:
  - Telugu (తెలుగు)
  - Hindi (हिंदी)
  - Tamil (தமிழ்)
  - Kannada (కన్నడ)
- **Google Translate Integration**: Automatic translation with error handling
- **Performance Metrics**: Execution time tracking

### Web Interface
- **Modern, Responsive Design**: Works on desktop and mobile devices
- **Real-time Processing Pipeline**: All modules execute sequentially
- **Multi-Format File Upload**: Supports PDF, DOCX, and TXT files
- **Dual Input Mode**: Type text directly or upload documents
- **Visual Stats Grid**: Displays key metrics (words found, noise removed, processing time)
- **Color-coded Module Sections**: Intuitive navigation between pipeline stages
- **Performance Metrics**: Real-time execution time tracking

---

## Architecture

```
T5 Neural Summarizer
│
├─ Module 0: Data Acquisition
│  └─ Text input & summary length configuration
│
├─ Module 1: Linguistic Analytics
│  ├─ Text cleaning (regex-based normalization)
│  ├─ spaCy NLP processing (POS tagging)
│  └─ Statistics generation
│
├─ Module 2: T5 Summarization
│  ├─ Tokenization (T5Tokenizer)
│  ├─ Model inference (T5ForConditionalGeneration)
│  └─ Beam search decoding
│
└─ Module 3: Multilingual Translation
   ├─ Language selection
   └─ Google Translate API
```

---

## Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (recommended for model loading)
- Internet connection (for model downloads & translation API)

### Dependencies
```
Flask
transformers
torch
spacy
nltk
deep-translator
PyPDF2
python-docx
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/BandlapalliBhanutejareddy/nlp-cpstone-.git
cd nlp-cpstone-
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Required Packages
```bash
pip install flask transformers torch nltk spacy deep-translator PyPDF2 python-docx
```

Or use the requirements file (if available):
```bash
pip install -r requirements.txt
```

### 4. Download NLP Models
The application automatically downloads required models on first run:
```bash
# Manual download (optional - happens automatically)
python -m spacy download en_core_web_sm
```

### 4. Download NLP Models
The application automatically downloads required models on first run:
```bash
# Manual download (optional - happens automatically)
python -m spacy download en_core_web_sm
```

### 5. Run the Application
```bash
python app.py
```

The application will start at `http://localhost:5000`

---

## Quick Start

1. **Clone & Setup** (2 minutes):
   ```bash
   git clone https://github.com/BandlapalliBhanutejareddy/nlp-cpstone-.git
   cd nlp-cpstone-
   python -m venv venv
   venv\Scripts\activate
   pip install flask transformers torch nltk spacy deep-translator PyPDF2 python-docx
   ```

2. **Run the App**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   - Navigate to `http://localhost:5000`
   - Paste your text and click "Execute All Modules"

---

## Usage

### Web Interface Usage

1. **Enter Text** (Module 0)
   - Paste or type the text you want to summarize in the textarea
   - Set the desired number of summary lines
   - Click "Execute All Modules"

2. **View Analysis Results** (Module 1)
   - See original word count, cleaned word count, and noise reduction
   - Review Part-of-Speech analysis with color-coded badges
   - Check the cleaned text output

3. **Review Summary** (Module 2)
   - Read the generated abstractive summary
   - Module 2 execution time is displayed

4. **Translate Summary** (Module 3)
   - Select target language from dropdown
   - Click "Translate Summary"
   - View translated output with timing

### API Endpoints

#### Process Pipeline Endpoint
**POST** `/process`

Request body:
```json
{
  "text": "Your long text here...",
  "lines": 5
}
```

Response:
```json
{
  "cleaned": "cleaned text...",
  "stats": {
    "word_count": 100,
    "cleaned_word_count": 95,
    "sentence_count": 8,
    "reduction": "5%"
  },
  "pos": [
    {"text": "word", "pos": "NOUN"},
    ...
  ],
  "summary": "Generated summary...",
  "times": {
    "module1": 0.234,
    "module2": 1.567
  }
}
```

#### Translation Endpoint
**POST** `/translate`

Request body:
```json
{
  "summary": "Text to translate...",
  "lang": "te"  // or "hi", "ta", "kn"
}
```

Response:
```json
{
  "translated": "Translation...",
  "time": 0.456
}
```

---

## Project Structure

```
nlp-cpstone-/
├── app.py                 # Flask application & API routes
├── test_t5.py             # Testing script for T5 model
├── README.md              # This file
├── .gitignore             # Git ignore file (excludes venv/)
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Directory for uploaded documents
│   ├── *.pdf              # Sample PDF files
│   └── *.docx             # Sample Word documents
└── venv/                  # Virtual environment (not committed to repo)
```

**Note**: The `venv/` directory and `README.md` are excluded from git tracking via `.gitignore`.

---

## Technical Details

### Module 1: Linguistic Analysis
- **Text Cleaning**: Regex patterns remove non-alphanumeric characters except punctuation
- **POS Tagging**: Uses spaCy's trained English model for grammatical analysis
- **Tokenization**: NLTK used for sentence and word tokenization

### Module 2: T5 Summarization
- **Model**: `t5-small` (60M parameters) - Pre-trained on 750GB of text (C4 dataset)
- **Task Format**: Prefix-based task specification (`"summarize: " + text`)
- **Input Tokenization**: Max token limit of 1024 (automatic truncation)
- **Decoding Parameters**:
  - `max_length`: 400 tokens
  - `min_length`: 120 tokens
  - `num_beams`: 3 (beam search - optimized for speed/quality)
  - `no_repeat_ngram_size`: 3 (prevents n-gram repetition)
  - `length_penalty`: 2.0 (controls output length)
  - `early_stopping`: True (stops when redundant patterns detected)
- **Output**: High-quality abstractive summaries with reduced hallucination

### Module 3: Translation
- **Provider**: Google Translate API (via deep-translator library)
- **Mode**: Auto-detection of source language
- **Error Handling**: Returns error message if translation fails

---

## Example Workflow

**Input Text**:
```
Artificial Intelligence has revolutionized multiple industries. 
It powers recommendation systems, autonomous vehicles, and medical diagnostics. 
Machine learning models learn patterns from data without explicit programming. 
Deep learning uses neural networks with multiple layers for complex pattern recognition.
```

**Module 1 Output**:
- Original Words: 45
- Cleaned Words: 44  
- Reduction: 2%
- POS Tags: AI (NOUN), revolutionized (VERB), industries (NOUN), etc.

**Module 2 Output**:
```
Artificial Intelligence has transformed various sectors through machine learning 
and deep neural networks for pattern recognition and decision-making.
```

**Module 3 Output (Telugu)**:
```
ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ మెషిన్ లర్నింగ్ మరియు డీప్ న్యూరల్ నెట్‌వర్క్‌ల 
ద్వారా విభిన్న రంగాలను పరిణామం చేసింది.
```

---

## Configuration & Customization

### Adjusting Summarization Behavior
Edit `app.py` to modify T5 generation parameters:

```python
outputs = t5_model.generate(
    inputs, 
    max_length=600,      # Increase for longer summaries
    min_length=200,      # Decrease for shorter summaries
    num_beams=6,         # Increase for better quality (slower)
    length_penalty=4.0   # Adjust to favor longer/shorter output
)
```

### Adding Languages
Modify the language dropdown in `templates/index.html`:

```html
<select id="langSelect">
    <option value="es">Spanish (Español)</option>
    <option value="fr">French (Français)</option>
    <!-- Language codes: ISO 639-1 standard -->
</select>
```

### Using Larger T5 Model
For better quality at computational cost:

```python
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")  # 220M params
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError" for transformers or torch
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: spaCy model not found
**Solution**: Manually download the model
```bash
python -m spacy download en_core_web_sm
```

### Issue: Slow inference on first run
**Solution**: This is normal - models cache on first load. Subsequent runs are much faster.

### Issue: Translation fails
**Solution**: Check internet connection. Google Translate API may have rate limits.

### Issue: Out of memory errors
**Solution**: 
- Use smaller model: `t5-small` instead of `t5-base`
- Reduce `max_length` parameter
- Restart Flask application to clear memory

---

## Performance Metrics

Typical execution times on standard hardware (Intel i7, 8GB RAM):

| Module | Task | Time | Hardware |
|--------|------|------|----------|
| Module 1 | Clean & Analyze & Tokenize | 0.1-0.3s | CPU |
| Module 2 | T5 Summarization (3-beam) | 1-2s | CPU/GPU |
| Module 3 | Translate (4 languages) | 0.5-1.5s | API |
| **Total** | **Complete Pipeline** | **1.5-4s** | Typical |

*Note: First run includes model initialization (~30s). Subsequent runs are faster with cached models.*

**Performance Tip**: Use GPU acceleration for faster inference on Module 2.

---

## NLP Techniques Used

### Core NLP Techniques

1. **Text Tokenization**
   - Word tokenization using NLTK `word_tokenize()`
   - Sentence tokenization using NLTK `sent_tokenize()`
   - Subword tokenization via T5Tokenizer for transformer compatibility

2. **Text Preprocessing & Noise Reduction**
   - Regex-based pattern matching to identify non-alphanumeric characters
   - Whitespace normalization using regex
   - Noise percentage calculation for text quality assessment

3. **Transformer-Based Abstractive Summarization**
   - **Architecture**: T5 (Text-to-Text Transfer Transformer)
   - **Pre-training**: Transfer learning using C4 corpus (750GB cleaned text)
   - **Task Format**: Prefix-based task specification (`"summarize: " + text`)
   - **Inference**: Encoder-decoder architecture with `T5ForConditionalGeneration`

4. **Advanced Decoding Strategies**
   - **Beam Search**: Explores multiple hypothesis sequences (3 beams)
   - **N-gram Blocking**: Prevents repetitive patterns with `no_repeat_ngram_size=3`
   - **Length Penalty**: Controls output length with penalty factor 2.0
   - **Early Stopping**: Terminates when sequences become redundant

5. **Multilingual Neural Machine Translation**
   - Language auto-detection
   - Google Translate API integration
   - Support for 4 regional languages (Telugu, Hindi, Tamil, Kannada)

6. **Document Processing**
   - PDF text extraction using PyPDF2
   - DOCX parsing using python-docx
   - Multi-format input support (PDF, DOCX, TXT)

### Pipeline Architecture

```
Raw Text/Document
    ↓
[Module 1: Tokenization & Cleaning]
    ├─ Word tokenization
    ├─ Noise detection & removal
    └─ Whitespace normalization
    ↓
[Module 2: T5 Summarization]
    ├─ T5Tokenizer (subword tokenization)
    ├─ T5 Encoder (768-dim embeddings)
    ├─ T5 Decoder with Beam Search
    └─ Early stopping & length control
    ↓
[Module 3: Translation]
    ├─ Language detection (auto)
    └─ Neural Machine Translation
    ↓
Final Output (Summary + Translation)
```

---

## Future Enhancements

- [ ] Support for additional languages
- [ ] Custom T5 fine-tuning for domain-specific summarization
- [ ] Batch processing for multiple documents
- [ ] Advanced POS visualization with dependency parsing
- [ ] Export results to PDF/JSON formats
- [ ] User session management and history
- [ ] Model switching (BART, Pegasus alternatives)
- [ ] GPU acceleration support

---

## License

This project is provided as-is for educational and research purposes.

---

## Contributions

Contributions, bug reports, and feature suggestions are welcome!

---

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [T5 Paper: Exploring the Limits of Transfer Learning](https://ai.google/research/pubs/T5)
- [spaCy NLP Library](https://spacy.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Version**: 1.0.0  
**Last Updated**: February 8, 2026  
**Repository**: [GitHub - nlp-cpstone-](https://github.com/BandlapalliBhanutejareddy/nlp-cpstone-)
# nlp-cpstone-
