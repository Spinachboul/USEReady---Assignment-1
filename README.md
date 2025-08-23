# Contract NER Extraction

A machine learning–driven **Named Entity Recognition (NER)** system for extracting key fields from legal contracts.
No regex, no rules — powered by **HuggingFace Transformers** fine-tuned for contract entity extraction.

---

## Features

* Train a **BERT-based token classification model** on annotated contracts
* Extract fields like:

  * Agreement Value
  * Agreement Start Date
  * Agreement End Date
  * Renewal Notice (Days)
  * Party One
  * Party Two
* Evaluate precision, recall, and F1 on held-out test contracts
* Run inference on a single contract (txt, docx, pdf)
* Interactive **Streamlit app** for uploading contracts & visualizing extracted entities

---

## Project Structure

```
contract-ner/
│── app.py                  # Streamlit app
│── train_model.py          # Train HuggingFace NER model
│── evaluate_model.py       # Evaluate model on test set
│── inference.py            # Extract entities from a single document
│── requirements.txt        # Dependencies
│
├── utils/
│   ├── data_loader.py      # Load docs (txt, pdf, docx)
│   ├── metrics.py          # Precision, recall, F1
│   └── preprocessing.py    # Tokenization, dataset prep (BIO tagging)
│
├── data/
│   ├── train.csv           # Training metadata
│   ├── test.csv            # Test metadata
│   ├── train/              # Train docs
│   └── test/               # Test docs
│
└── contract-ner-model/     # Saved HuggingFace model after training
```

---

## Installation

```bash
git clone https://github.com/yourname/useREADY-Assignment-1.git
cd contract-ner
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

---

## Training

Train the model on your dataset:

```bash
python -m train_model.py
```

The trained model will be saved in:

```
contract-ner-model/
```

---

## Evaluation

Evaluate model performance on the test set:

```bash
python -m evaluate_model.py
```

Outputs per-field and overall precision, recall, and F1.

---

## Streamlit App

Launch the interactive app:

```bash
streamlit run app.py
```

Features:

* Upload a contract → Extract fields
* Upload a `test.csv` → View evaluation metrics

## Requirements

* transformers
* datasets
* torch
* scikit-learn
* pandas
* numpy
* streamlit
* python-docx
* PyPDF2

(Installed via `requirements.txt`)

👉 Do you want me to also add a **usage example screenshot** (Streamlit UI + extracted JSON) to the README for extra clarity?
