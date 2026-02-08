from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
from deep_translator import GoogleTranslator
import PyPDF2
import docx
import nltk
import re
import os
import time

# Initialize NLTK
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MODULE 1: FAST CLEANING & NOISE ANALYTICS ---
def analyze_and_clean(raw_text):
    start_time = time.time()
    
    # Identify non-alphanumeric "noise" characters
    noise_pattern = r'[^a-zA-Z0-9\s.,!?]'
    noise_chars = re.findall(noise_pattern, raw_text)
    
    # Calculate Noise Percentage
    noise_found = len(noise_chars)
    total_chars = max(1, len(raw_text))
    noise_percent = round((noise_found / total_chars) * 100, 2)
    
    # Clean text: remove noise and normalize spaces
    cleaned = re.sub(noise_pattern, '', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    stats = {
        "original_words": len(word_tokenize(raw_text)),
        "cleaned_words": len(word_tokenize(cleaned)),
        "noise_found": noise_found,
        "noise_percentage": f"{noise_percent}%",
        "exec_time": round(time.time() - start_time, 4)
    }
    return cleaned, stats

# --- MODULE 2: OPTIMIZED T5 SUMMARIZER ---
print("Initializing Optimized T5 Engine...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

def fast_summarize(text):
    input_text = "summarize: " + text
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Optimized for speed: Reduced beams and enabled early stopping
    outputs = t5_model.generate(
        inputs, 
        max_length=400, 
        min_length=120, 
        num_beams=3,        # Faster than 5 or 6
        length_penalty=2.0, 
        no_repeat_ngram_size=3,
        early_stopping=True # Stops generation when tokens become redundant
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- ROUTES ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_data():
    text = ""
    # Dual-Input Check
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(filepath)
            text = " ".join([p.extract_text() or "" for p in reader.pages])
        elif filename.endswith('.docx'):
            doc = docx.Document(filepath)
            text = " ".join([p.text for p in doc.paragraphs])
        else:
            text = file.read().decode('utf-8')
    else:
        text = request.form.get("text", "")

    if not text.strip():
        return jsonify({"error": "Input is empty"})

    cleaned, stats = analyze_and_clean(text)
    summary = fast_summarize(cleaned)
    
    return jsonify({"cleaned": cleaned, "stats": stats, "summary": summary})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    try:
        translated = GoogleTranslator(source='auto', target=data['lang']).translate(data['summary'])
        return jsonify({"translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)