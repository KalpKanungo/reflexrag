from datasets import load_dataset
import json

dataset = load_dataset("microsoft/ms_marco","v1.1")

train = dataset["train"]

pairs = []

for i in range(30000):
    q = train[i]["query"]
    passages = train[i]["passages"]["passage_text"]
    pos = passages[0]
    pairs.append({"query":q,"passage":pos,"label":1})

with open("training/msmarco_train.json","w") as f:
    json.dump(pairs,f)