from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import json
import torch
import torch.nn as nn
from datasets import Dataset, Value
from transformers import DataCollatorWithPadding

model_name = "BAAI/bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # ← fixed
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

dataset = dataset.map(tokenize, remove_columns=["query", "passage", "label"])
dataset = dataset.cast_column("labels", Value("float32"))  # ← fixed

# Custom Trainer with BCEWithLogitsLoss
class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="models/reranker_finetuned",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50,
    save_strategy="no",        # ← no checkpoints saved
)

trainer = RerankerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("models/reranker_finetuned")