# Fine tunning BERT AGENT 1

import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import re
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    XLMRobertaConfig,
    DataCollatorForTokenClassification
)
import seqeval
from huggingface_hub import login

## Autentification HF

login(token="YOUR_HF_TOKEN_HERE")


## Data Load and Preprocessing

dataset_path = "PATH_TO_YOUR_DATA"

with open(dataset_path, "r", encoding="utf-8") as f:
    ner_dataset = json.load(f)

print(f"Loaded {len(ner_dataset)} clinical notes")

ner_dataset[0]


TOKEN_RE = re.compile(r'\w+|[^\w\s]')

def tokenize_with_offsets(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

def process_ner_dataset(ner_dataset):
    processed_data = []
    unique_labels = set()
    
    for item in ner_dataset:
        text = item["data"]["comment"]
        tokens_data = tokenize_with_offsets(text)
        tokens = [t[0] for t in tokens_data]
        
        # extract annotations type label
        spans = []
        if not item.get("annotations"):
            continue
            
        for r in item["annotations"][0]["result"]:
            if r["type"] == "labels":
                v = r["value"]
                label = v["labels"][0]
                spans.append({
                    "start": v["start"],
                    "end": v["end"],
                    "label": label
                })
                unique_labels.add(label)
        
        # token assignation
        labels = ["O"] * len(tokens)
        for i, (word, t_start, t_end) in enumerate(tokens_data):
            for span in spans:
                if max(t_start, span["start"]) < min(t_end, span["end"]):
                    labels[i] = span["label"]
                    break
        
        processed_data.append({
            "tokens": tokens,
            "labels": labels
        })
        
    return processed_data, sorted(list(unique_labels))

# obtain final prepared samples
final_samples, label_list = process_ner_dataset(ner_dataset)

# --- DEBUG 1 ---
print("\n--- DEBUG 1 ---")
sample_idx = 700
debug_sample = final_samples[sample_idx]
print(f"Texto original (tokens): {debug_sample['tokens'][:20]}")
print(f"Labels asignadas: {debug_sample['labels'][:20]}")

entidades_encontradas = [l for l in debug_sample['labels'] if l != "O"]
print(f"Total tokens con etiqueta médica en esta muestra: {len(entidades_encontradas)}")

# mapping with ids
if "O" in label_list: label_list.remove("O")
label_list = ["O"] + label_list
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Procesados {len(final_samples)} ejemplos.")
print(f"Etiquetas encontradas: {label_list}")

final_samples[700]


## Train-test Split

# seed for reproducibility
random.seed(43)

# mix the dataset
ner_dataset_shuffled = final_samples.copy()
random.shuffle(ner_dataset_shuffled)

# compute split indices
n_total = len(ner_dataset_shuffled)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val  # 10%

# split
train_set = ner_dataset_shuffled[:n_train]
val_set = ner_dataset_shuffled[n_train:n_train + n_val]
test_set = ner_dataset_shuffled[n_train + n_val:]

print(f"Total samples: {n_total}")
print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")


## Fine Tune mBERT

dataset_dict = DatasetDict({
    "train": Dataset.from_list(train_set),
    "validation": Dataset.from_list(val_set),
    "test": Dataset.from_list(test_set)
})

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True, 
        max_length=512
    )

    labels = []
    for i, label_list in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                # Aquí label_list[word_idx] es un string, lo pasamos a ID
                label_ids.append(label2id[label_list[word_idx]])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# MODEL LOADING

model_id = "FacebookAI/xlm-roberta-base"
# add_prefix_space is essential for RoBERTa-based models when passing word lists
tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

tokenized_ds = dataset_dict.map(
    tokenize_and_align_labels, 
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)

# --- DEBUG 2 ---
print("\n--- DEBUG 2 ---")
test_idx = 10
example = tokenized_ds["train"][test_idx]

deberta_tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
labels_with_ids = example["labels"]

print(f"{'Sub-Token':<20} | {'Label ID':<10} | {'Label Name'}")
print("-" * 45)
for t, l_id in zip(deberta_tokens[:30], labels_with_ids[:30]):
    l_name = id2label[l_id] if l_id != -100 else "IGNORAR (-100)"
    print(f"{t:<20} | {l_id:<10} | {l_name}")

config = XLMRobertaConfig.from_pretrained(
    model_id, 
    num_labels=len(label_list), 
    id2label=id2label, 
    label2id=label2id
)

model = AutoModelForTokenClassification.from_pretrained(model_id, config=config)


metric = evaluate.load("seqeval")

# metrics to print in each training
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Aplanamos y filtramos los -100
    y_true = []
    y_pred = []
    for prediction, label in zip(predictions, labels):
        for p_val, l_val in zip(prediction, label):
            if l_val != -100:
                y_true.append(l_val)
                y_pred.append(p_val)
    # DEBUG 3
    unique_preds, counts_preds = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip([id2label[i] for i in unique_preds], counts_preds))
    print(f"\n[Epoch Debug] Model predictions: {pred_dist}")

    # global metrics per token
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

data_collator = DataCollatorForTokenClassification(tokenizer)


training_args = TrainingArguments(
    output_dir="./RESULTS_BERT_V1",
    eval_strategy="epoch",
    learning_rate=2e-5,           # XLM-R can handle a slightly higher LR than DeBERTa
    lr_scheduler_type="linear",   # Ensures smooth transitions
    warmup_ratio=0.1,             # Warm up for the first 10% of steps
    per_device_train_batch_size=16, 
    num_train_epochs=10,
    weight_decay=0.01,            # Helps prevent overfitting
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True if torch.cuda.is_available() else False, # Use fp16 for speed if on GPU
    report_to="none" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

batch = data_collator([tokenized_ds["train"][i] for i in range(5)])

labels_planas = batch['labels'].flatten()
entidades_reales = torch.sum((labels_planas > 0)).item()
tokens_ignorados = torch.sum((labels_planas == -100)).item()
tokens_o = torch.sum((labels_planas == 0)).item()


trainer.train()


## Get Results

def final_classification_report(trainer, dataset):
    
    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=2)
    labels = output.label_ids

    y_true = []
    y_pred = []
    results_list = []

    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        mask = label != -100
        true_label = [id2label[l] for l in label[mask]]
        pred_label = [id2label[p] for p, l in zip(prediction, label) if l != -100]
        
        y_true.extend(true_label)
        y_pred.extend(pred_label)
        
        results_list.append({
            "sample_index": i,
            "true": true_label,
            "pred": pred_label
        })

    report = classification_report(y_true, y_pred, digits=3)
    
    labels_ordered = [l for l in label_list if l != "O"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_ordered)
    
    cm_df = pd.DataFrame(cm, index=labels_ordered, columns=labels_ordered)
    
    return results_list, report, cm_df


results, report, conf_matrix = final_classification_report(trainer, tokenized_ds["test"])



output_path_json = "cr-multia/medical_ner/FINETUNNING/RESULTS_BERT_V1/BERT_results.json"
with open(output_path_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_path_json}")

output_path_txt = "cr-multia/medical_ner/FINETUNNING/RESULTS_BERT_V1/medical_ner_report_BERT_FINETUNNED.txt"
with open(output_path_txt, "w") as f:
    f.write("Medical NER Classification Report\n")
    f.write(report)

print(f"Report saved to {output_path_txt}")

def save_confusion_matrix(cm_df, output_path="confusion_matrix_roberta_v1.png"):
    plt.figure(figsize=(12, 10))

    sns.set_theme(style="white")

    plot = sns.heatmap(
        cm_df,
	    annot=True,   
        fmt='d',       
        cmap='Blues',  
        cbar=True,	
        linewidths=.5   
    )

    plt.title('Confusion Matrix - Agent 1 XLM-RoBERTa', fontsize=15)
    plt.ylabel('Real Label', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Confusion Matrix saved in: {output_path}")

    plt.show()

save_confusion_matrix(conf_matrix, "confusion_matrix_roberta_v1.png")

