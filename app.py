from flask import Flask, render_template, request, redirect, send_file
import nltk, io, json, tempfile
import docx2txt, PyPDF2
from googletrans import Translator
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk.tree import Tree
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

app = Flask(__name__)
translator = Translator()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

DATA = {
    "text": "",
    "sentences": [],
    "summary": "",
    "translated_langs": {},
    "pos_groups": {},
    "pos_labels": [],
    "pos_values": [],
    "top_sentence": "",
    "parse_tree": {}
}

# ---------- TEXT EXTRACTION ----------
def extract_text(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.filename.endswith(".docx"):
        return docx2txt.process(file)
    return ""

# ---------- SENTENCE IMPORTANCE ----------
def analyze_sentences(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    max_score = max(scores) if scores.any() else 1

    stop_words = set(stopwords.words("english"))
    result = []
    for i, s in enumerate(sentences):
        score = round((scores[i] / max_score) * 100, 2)
        words = word_tokenize(s.lower())
        keywords = [w for w in words if w.isalpha() and w not in stop_words]
        result.append({
            "sentence": s,
            "importance": score,
            "length": len(words),
            "keywords": list(set(keywords[:6]))
        })

    return sorted(result, key=lambda x: x["importance"], reverse=True)

# ---------- SUMMARY ----------
def summarize_from_sentences(sentences):
    return " ".join([s["sentence"] for s in sentences[:3]])

# ---------- POS ANALYSIS ----------
def analyze_pos(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    pos_groups = {}
    for w, t in tagged:
        pos_groups.setdefault(t, []).append(w)

    counts = Counter(t for _, t in tagged)
    return pos_groups, list(counts.keys()), list(counts.values())

# ---------- ENGLISH NER ----------
def extract_entities_english(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return list(set([w for w, t in tagged if t in ["NNP", "NNPS"]]))

# ---------- PARSE TREE ----------
def build_parse_tree(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    grammar = r"""
      NP: {<DT>?<JJ>*<NN.*>+}
      VP: {<VB.*><NP|PP>*}
      PP: {<IN><NP>}
      S: {<NP><VP>}
    """

    cp = RegexpParser(grammar)
    tree = cp.parse(tagged)

    def convert(t):
        if isinstance(t, Tree):
            return {"name": t.label(), "children": [convert(c) for c in t]}
        else:
            return {"name": t[0]}

    return convert(tree)

# ---------- HOME ----------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text", "")

        if 'file' in request.files and request.files['file'].filename != "":
            text = extract_text(request.files['file'])

        DATA["text"] = text
        DATA["sentences"] = analyze_sentences(text)
        DATA["summary"] = summarize_from_sentences(DATA["sentences"])

        DATA["pos_groups"], DATA["pos_labels"], DATA["pos_values"] = analyze_pos(text)

        if DATA["sentences"]:
            DATA["top_sentence"] = DATA["sentences"][0]["sentence"]
            DATA["parse_tree"] = build_parse_tree(DATA["top_sentence"])

        return redirect("/analysis")

    return render_template("module0.html")

# ---------- MODULE 1 ----------
@app.route("/analysis")
def analysis():
    tone = "Neutral"
    if DATA["text"]:
        p = TextBlob(DATA["text"]).sentiment.polarity
        if p > 0.1:
            tone = "Positive"
        elif p < -0.1:
            tone = "Negative"

    return render_template("module1.html",
                           sent_cards=DATA["sentences"],
                           tone=tone,
                           top_sentence=DATA["top_sentence"],
                           parse_tree=json.dumps(DATA["parse_tree"]))

# ---------- TRANSLATION ----------
@app.route("/translation", methods=["GET", "POST"])
def translation():
    sem_sim = None
    ner_entities = []
    translated_text = None
    selected_lang = None

    if request.method == "POST" and DATA["summary"]:
        selected_lang = request.form.get("lang")

        translated_text = translator.translate(DATA["summary"], dest=selected_lang).text
        DATA["translated_langs"][selected_lang] = translated_text

        vect = TfidfVectorizer().fit([DATA["summary"], translated_text])
        vecs = vect.transform([DATA["summary"], translated_text])
        sem_sim = round((1 - cosine_similarity(vecs[0], vecs[1])[0][0]) * 10, 2)

        ner_entities = extract_entities_english(DATA["summary"])

    return render_template("module2.html",
                           summary=DATA["summary"],
                           translated=translated_text,
                           sem_sim=sem_sim,
                           ner_entities=ner_entities,
                           selected_lang=selected_lang)

# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",
        pos_labels=DATA["pos_labels"],
        pos_values=DATA["pos_values"],
        pos_groups=DATA["pos_groups"]
    )

# ---------- DOWNLOAD REPORT ----------
@app.route("/download_report")
def download_report():
    buffer = io.BytesIO()

    # Load Unicode font (important)
    pdfmetrics.registerFont(TTFont('Noto', 'fonts/NotoSans-Regular.ttf'))

    styles = getSampleStyleSheet()
    styles['BodyText'].fontName = 'Noto'
    styles['Heading2'].fontName = 'Noto'

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Title
    elements.append(Paragraph(
        "Explainable Multilingual NLP Intelligence Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Original Text
    elements.append(Paragraph("<b>Original Text:</b>", styles['Heading2']))
    elements.append(Paragraph(DATA["text"], styles['BodyText']))
    elements.append(Spacer(1, 15))

    # Summary
    elements.append(Paragraph("<b>Generated Summary:</b>", styles['Heading2']))
    elements.append(Paragraph(DATA["summary"], styles['BodyText']))
    elements.append(Spacer(1, 20))

    # -------- POS GRAPH IMAGE --------
    if DATA["pos_labels"]:
        plt.figure(figsize=(7, 3))
        plt.bar(DATA["pos_labels"], DATA["pos_values"])
        plt.title("POS Tag Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()

        pos_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(pos_img.name, dpi=200)
        plt.close()

        elements.append(Paragraph("<b>POS Tag Distribution Graph:</b>", styles['Heading2']))
        elements.append(Image(pos_img.name, width=480, height=220))
        elements.append(Spacer(1, 20))

    # -------- POS TAGSET WITH WORDS --------
    elements.append(Paragraph("<b>POS Tags with Important Words:</b>", styles['Heading2']))
    for tag, words in DATA["pos_groups"].items():
        line = f"{tag}: {', '.join(words[:10])}"
        elements.append(Paragraph(line, styles['BodyText']))
    elements.append(Spacer(1, 20))

    # -------- PARSE TREE IMAGE --------
    if DATA["top_sentence"]:
        elements.append(Paragraph("<b>Highest Importance Sentence:</b>", styles['Heading2']))
        elements.append(Paragraph(DATA["top_sentence"], styles['BodyText']))
        elements.append(Spacer(1, 12))

        def draw_tree(node, x=0, y=0, dx=2.0, dy=1.5):
            plt.text(x, y, node["name"], ha='center')
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    new_x = x + (i - len(node["children"]) / 2) * dx
                    new_y = y - dy
                    plt.plot([x, new_x], [y, new_y])
                    draw_tree(child, new_x, new_y, dx/1.5, dy)

        plt.figure(figsize=(10, 5))
        draw_tree(DATA["parse_tree"], 0, 0)
        plt.axis('off')

        tree_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tree_img.name, dpi=200, bbox_inches='tight')
        plt.close()

        elements.append(Paragraph("<b>Parse Tree Diagram:</b>", styles['Heading2']))
        elements.append(Image(tree_img.name, width=500, height=260))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer,
                     as_attachment=True,
                     download_name="Explainable_NLP_Report.pdf",
                     mimetype="application/pdf")
if __name__ == "__main__":
    app.run(debug=True)