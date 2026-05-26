#!/usr/bin/env python
# coding: utf-8

# # Fine tunning LLMs: LLAMA3

# dependencies
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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    XLMRobertaConfig,
    DataCollatorForTokenClassification
)
import seqeval
from huggingface_hub import login

# login to hugging face
login(token="YOUR_HF_TOKEN_HERE")


# ## 1. Data Load and Preprocessing

dataset_path = "PATH_TO_YOUR_DATA"

with open(dataset_path, "r", encoding="utf-8") as f:
    ner_dataset = json.load(f)

print(f"Loaded {len(ner_dataset)} clinical notes")


ner_dataset[0]

# here we need the mapping of the labels to the ids 

unique_attributes = ["Confirmed", "Control", "Progression", "Suspicion", "Discarded", "Yes", "Previous", "No"]
label2id = {label: i for i, label in enumerate(unique_attributes)}
id2label = {i: label for i, label in enumerate(unique_attributes)}

# prepare dataset

ALLOWED_LABELS = ["Diagnosis", "Smoker", "GeneMutation", "Treatment", "Exitus", "FamilyHistory"]

def prepare_attribute_data(dataset):
    rows = []
    for item in dataset:
        text = item["data"]["comment"]
        annotations = item.get("annotations", [{}])[0].get("result", [])
        
        id_to_label = {}
        id_to_choice = {}
        
        for r in annotations:
            if r["type"] == "labels":
                id_to_label[r["id"]] = {
                    "text": r["value"]["text"],
                    "label": r["value"]["labels"][0],
                    "start": r["value"]["start"],
                    "end": r["value"]["end"]
                }
            elif r["type"] == "choices":
                id_to_choice[r["id"]] = r["value"]["choices"][0]

        for region_id, info in id_to_label.items():
            if info["label"] not in ALLOWED_LABELS:
                continue
            
            # Formatear contexto igual que en inferencia
            full_context = text[:info['start']] + f"[{info['text']}]" + text[info['end']:]
            true_attr = id_to_choice.get(region_id, "None")
            
            if true_attr != "None":
                rows.append({
                    "context": full_context,
                    "entity_text": info["text"],
                    "entity_label": info["label"],
                    "attribute": true_attr
                })
    return rows

# train test val split 

random.seed(42)
all_samples = prepare_attribute_data(ner_dataset)
ner_dataset_shuffled = all_samples.copy()
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


# ## 3. Fine Tune Llama3


# fine tune with qlora llama3 8B
model_id = "FacebookAI/xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

def preprocess_function(examples):
    inputs = tokenizer(
        examples["context"],
        examples["entity_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    inputs["labels"] = [label2id[attr] for attr in examples["attribute"]]
    return inputs

ds_train = Dataset.from_list(train_set).map(preprocess_function, batched=True)
ds_val = Dataset.from_list(val_set).map(preprocess_function, batched=True)
ds_test = Dataset.from_list(test_set).map(preprocess_function, batched=True)

# RoBERTa needs PyTorch tensors
ds_train.set_format("torch")
ds_val.set_format("torch")
ds_test.set_format("torch")

columns_to_remove = ["context", "entity_text", "entity_label", "attribute"]

ds_train = ds_train.remove_columns(columns_to_remove)
ds_test = ds_test.remove_columns(columns_to_remove)
ds_val = ds_val.remove_columns(columns_to_remove)

# load model and trainer
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=len(unique_attributes),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    y_true = labels
    y_pred = predictions

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

training_args = TrainingArguments(
    output_dir="./RESULTS_BERT",
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
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
)

trainer.train()

# save adapter
OUT_DIR_LLM = "cr-multia/attribute_association/FINETUNNING"
out = trainer.save_model(OUT_DIR_LLM)
out
print("Adapter would be saved to:", OUT_DIR_LLM)


# ## 4. Get Results

def final_classification_report_attr(trainer, dataset, id2label, label_list):
    output = trainer.predict(dataset)
    y_pred_ids = np.argmax(output.predictions, axis=1)
    y_true_ids = output.label_ids

    y_true = [id2label[l] for l in y_true_ids]
    y_pred = [id2label[p] for p in y_pred_ids]

    results_list = []
    for i in range(len(y_true)):
        results_list.append({
            "sample_index": i,
            "true": y_true[i],
            "pred": y_pred[i]
        })

    report = classification_report(y_true, y_pred, digits=3)

    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)

    return results_list, report, cm_df

results, report, cm = final_classification_report_attr(trainer, ds_test, id2label, unique_attributes)

# save results
output_path = Path("cr-multia/attribute_association/FINETUNNING")
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / "bert_results_v1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

with open(output_path / "attr_report_BERT_FINETUNNED_v1.txt", "w") as f:
    f.write(report)

def save_confusion_matrix(cm_df, output_path="confusion_matrix_roberta_v1.png"):
    plt.figure(figsize=(12, 10))
    
    sns.set_theme(style="white")
    
    plot = sns.heatmap(
        cm_df, 
        annot=True,     # Pone los números en las celdas
        fmt='d',        # Formato de número entero
        cmap='Blues',   # Color azul (muy estándar en papers)
        cbar=True,      # Barra de color lateral
        linewidths=.5   # Líneas finas entre celdas
    )
    
    plt.title('Confusion Matrix - Agent 2 XLM-RoBERTa', fontsize=15)
    plt.ylabel('Real Attribute', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300) 
    print(f"Confusion Matrix daved in: {output_path}")
    
    plt.show()

save_confusion_matrix(cm, "confusion_matrix_roberta_v1.png")

