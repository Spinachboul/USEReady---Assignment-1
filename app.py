# app.py
import os
import tempfile
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

# =====================
# Load Model
# =====================
@st.cache_resource
def load_model():
    model_dir = "contract-ner-model"  # path to your trained model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner_pipeline = load_model()

# =====================
# File Readers
# =====================
def read_file(path):
    ext = path.split(".")[-1].lower()
    if ext == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == "docx":
        from docx import Document
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "pdf":
        import PyPDF2
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    return ""

# =====================
# Evaluation
# =====================
def evaluate(test_csv, test_dir):
    df_test = pd.read_csv(test_csv)

    fields = [
        "Agreement Value",
        "Agreement Start Date",
        "Agreement End Date",
        "Renewal Notice (Days)",
        "Party One",
        "Party Two"
    ]
    results = {f: {"y_true": [], "y_pred": []} for f in fields}

    for _, row in df_test.iterrows():
        fname = row["File Name"]
        filepath = None
        for ext in [".txt", ".docx", ".pdf"]:
            candidate = os.path.join(test_dir, fname + ext)
            if os.path.exists(candidate):
                filepath = candidate
                break
        if not filepath:
            continue

        text = read_file(filepath)
        ents = ner_pipeline(text)

        preds = {}
        for ent in ents:
            preds[ent["entity_group"]] = ent["word"]

        for field in fields:
            true_val = str(row.get(field, "")).strip().lower()
            pred_val = str(preds.get(field.upper().replace(" ", "_"), "")).strip().lower()

            results[field]["y_true"].append(1 if true_val else 0)
            results[field]["y_pred"].append(1 if pred_val else 0)

    # Compute per-field metrics
    metrics = []
    overall_true, overall_pred = [], []

    for field in fields:
        y_true = results[field]["y_true"]
        y_pred = results[field]["y_pred"]

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        overall_true.extend(y_true)
        overall_pred.extend(y_pred)

        metrics.append({
            "Field": field,
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1-score": round(f1, 2)
        })

    overall_prec = precision_score(overall_true, overall_pred, zero_division=0)
    overall_rec = recall_score(overall_true, overall_pred, zero_division=0)
    overall_f1 = f1_score(overall_true, overall_pred, zero_division=0)

    metrics.append({
        "Field": "Overall",
        "Precision": round(overall_prec, 2),
        "Recall": round(overall_rec, 2),
        "F1-score": round(overall_f1, 2)
    })

    return pd.DataFrame(metrics)

# =====================
# Streamlit UI
# =====================
st.title("📑 Contract NER Evaluation")

st.markdown("Upload **test.csv** and the **test directory** (with TXT/DOCX/PDF files).")

test_csv = st.file_uploader("Upload test.csv", type=["csv"])
test_dir = st.file_uploader("Upload test directory (zipped)", type=["zip"])

if test_csv and test_dir:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save CSV
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, "wb") as f:
            f.write(test_csv.getbuffer())

        # Extract ZIP
        import zipfile
        with zipfile.ZipFile(test_dir, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find extracted folder
        extracted_folders = [os.path.join(tmpdir, d) for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        if extracted_folders:
            test_folder = extracted_folders[0]
        else:
            test_folder = tmpdir

        st.write("✅ Files uploaded successfully. Running evaluation...")

        metrics_df = evaluate(csv_path, test_folder)
        st.dataframe(metrics_df, use_container_width=True)
