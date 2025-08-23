import os
import pandas as pd
from datasets import Dataset
from utils.data_loader import read_file

def char_to_token_labels(text, spans, tokenizer, label2id):
    """
    Convert annotated spans (dict field_name -> string) into token-level BIO labels.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    labels = ["O"] * len(encoding["input_ids"])

    for field, value in spans.items():
        if not value or value.strip() == "":
            continue

        start_idx = text.lower().find(value.lower())
        if start_idx == -1:
            continue
        end_idx = start_idx + len(value)

        for i, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
            if tok_start >= end_idx or tok_end <= start_idx:
                continue
            prefix = "B-" if labels[i] == "O" else "I-"
            labels[i] = f"{prefix}{field.upper().replace(' ', '_')}"

    encoding["labels"] = [label2id.get(l, 0) for l in labels]
    return {k: v for k, v in encoding.items() if k != "offset_mapping"}  # drop offsets here


def prepare_hf_dataset(csv_path, docs_dir, tokenizer, label2id):
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        fname = row["File Name"]
        filepath = None
        for ext in [".txt", ".docx", ".pdf"]:
            candidate = os.path.join(docs_dir, fname + ext)
            if os.path.exists(candidate):
                filepath = candidate
                break
        if not filepath:
            continue

        text = read_file(filepath)
        labels = {col: str(row.get(col, "")).strip() for col in df.columns if col != "File Name"}
        records.append({"text": text, "labels": labels})

    dataset = Dataset.from_list(records)

    # Apply tokenization + alignment row by row
    dataset = dataset.map(
        lambda row: char_to_token_labels(row["text"], row["labels"], tokenizer, label2id),
        batched=False
    )

    return dataset   # no remove_columns
