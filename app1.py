import streamlit as st
import pandas as pd
from inference import extract_entities
from utils.data_loader import read_file
from evaluate_model import fields, compute_exact_match

st.set_page_config(page_title="Contract NER Extraction App", layout="wide")

st.title("📄 Contract NER Extraction App")

uploaded_doc = st.file_uploader("Upload a contract (txt, docx, pdf)", type=["txt", "docx", "pdf"])

uploaded_csv = st.file_uploader("Upload test.csv for evaluation", type=["csv"])

if uploaded_doc:
    with open("temp.docx", "wb") as f:
        f.write(uploaded_doc.getvalue())
    text = read_file("temp.docx")
    ents = extract_entities(text)

    st.subheader("🔍 Extracted Entities")
    st.json(ents)

if uploaded_csv:
    df_test = pd.read_csv(uploaded_csv)

    st.subheader("Evaluation Metrics")

    scores = compute_exact_match(df_test, df_test)  

    for field in fields:
        st.markdown(f"**{field}:**")
        st.write(f"- Precision: {scores[field]['precision']:.2f}")
        st.write(f"- Recall: {scores[field]['recall']:.2f}")
        st.write(f"- F1-score: {scores[field]['f1']:.2f}")

    st.markdown("### Overall Metrics")
    st.write(f"- Precision: {scores['overall']['precision']:.2f}")
    st.write(f"- Recall: {scores['overall']['recall']:.2f}")
    st.write(f"- F1-score: {scores['overall']['f1']:.2f}")
