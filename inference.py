from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_pipeline():
    model_dir = "contract-ner-model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    ner_pipeline = load_pipeline()
    ents = ner_pipeline(text)
    preds = {}
    for ent in ents:
        preds[ent["entity_group"]] = ent["word"]
    return preds
