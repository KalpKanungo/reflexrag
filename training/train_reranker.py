from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import json
from datasets import Dataset
from transformers import DataCollatorWithPadding

model_name = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

data = json.load(open("training/msmarco_train.json"))

dataset = Dataset.from_list(data)

def tokenize(example):
    tokens = tokenizer(
        example["query"],
        example["passage"],
        truncation=True,
        padding=False,
        max_length=256
    )
    tokens["labels"] = float(example["label"])
    return tokens

dataset = dataset.map(tokenize, remove_columns=["query","passage","label"])

training_args = TrainingArguments(
    output_dir="models/reranker_finetuned",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("models/reranker_finetuned")