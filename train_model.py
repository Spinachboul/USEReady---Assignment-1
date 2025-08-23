from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from utils.preprocessing import prepare_hf_dataset

MODEL_NAME = "bert-base-uncased"

label_list = [
    "O",
    "B-AGREEMENT_VALUE", "I-AGREEMENT_VALUE",
    "B-AGREEMENT_START_DATE", "I-AGREEMENT_START_DATE",
    "B-AGREEMENT_END_DATE", "I-AGREEMENT_END_DATE",
    "B-RENEWAL_NOTICE_(DAYS)", "I-RENEWAL_NOTICE_(DAYS)",
    "B-PARTY_ONE", "I-PARTY_ONE",
    "B-PARTY_TWO", "I-PARTY_TWO",
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = prepare_hf_dataset("data/train.csv", "data/train", tokenizer, label2id)
test_dataset = prepare_hf_dataset("data/test.csv", "data/test", tokenizer, label2id)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir="contract-ner-model",
    # evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("contract-ner-model")
tokenizer.save_pretrained("contract-ner-model")
