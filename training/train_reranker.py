from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import json
import torch
from datasets import Dataset

model_name = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

data = json.load(open("training/msmarco_train.json"))

dataset = Dataset.from_list(data)

def tokenize(example):
    return tokenizer(example["query"],example["passage"],truncation=True,padding="max_length",max_length=512)

dataset = dataset.map(tokenize)

dataset = dataset.rename_column("label","labels")
dataset = dataset.cast_column("labels","float32")

training_args = TrainingArguments(
    output_dir="models/reranker_finetuned",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

trainer.save_model("models/reranker_finetuned")