#!/usr/bin/env python
# coding: utf-8

# # Fine tunning SLM

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
import torch.nn as nn
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
from collections import Counter
import unicodedata

# login to hugging face
login(token="YOUR_HF_TOKEN_HERE")


## 1. Data Load and Preprocessing

dataset_path = "PATH_TO_YOUR_DATA"

with open(dataset_path, "r", encoding="utf-8") as f:
    ner_dataset = json.load(f)

print(f"Loaded {len(ner_dataset)} clinical notes")


ner_dataset[0]

# 2. Label Definition

# here we need the mapping of the labels to the ids 
unique_attributes = ["Confirmed", "Control", "Progression", "Suspicion", "Discarded", "Yes", "Previous", "No"]
label2id = {label: i for i, label in enumerate(unique_attributes)}
id2label = {i: label for i, label in enumerate(unique_attributes)}

ALLOWED_LABELS = ["Diagnosis", "Smoker", "GeneMutation", "Treatment", "FamilyHistory"]

VALID_ATTRIBUTES_MAP = {
    "Diagnosis": ["Confirmed", "Suspicion", "Discarded", "Progression", "Control"],
    "Smoker": ["Yes", "Previous", "No"],
    "GeneMutation": ["Yes", "No"],
    "Treatment": ["Yes", "No"],
    "FamilyHistory": ["Yes", "No"]
}

# 3. Prepare Original Data

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

            valid_attrs = VALID_ATTRIBUTES_MAP.get(info["label"], [])
            if true_attr not in valid_attrs:
                continue

            if true_attr != "None":
                rows.append({
                    "context": full_context,
                    "entity_text": info["text"],
                    "entity_label": info["label"],
                    "attribute": true_attr
                })
    return rows

all_samples = prepare_attribute_data(ner_dataset)

# 4. Train-Test-Val Split

random.seed(42)
original_shuffled = all_samples.copy()
random.shuffle(original_shuffled)

n_total = len(original_shuffled)
n_train = int(n_total * 0.8)
n_val   = int(n_total * 0.1)

train_orig = original_shuffled[:n_train]
val_orig   = original_shuffled[n_train:n_train + n_val]
test_orig  = original_shuffled[n_train + n_val:]

print(f"Originales  — Train: {len(train_orig)}, Val: {len(val_orig)}, Test: {len(test_orig)}")

# 5. Augmentation - Only Training Samples

def normalize(text):
    """Elimina acentos manteniendo las letras base: és→es, à→a, ï→i"""
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

def augment_negation(context, entity_text):
    patterns = [
        # --- DIAGNOSIS / Discarded ---
        (r"no concluyente (de|para) \[{}\]",  ["se descarta [{}]", "no es confirma [{}]", "sin evidencia de [{}]"]),
        (r"no es confirma \[{}\]",            ["se descarta [{}]", "no concluyente de [{}]"]),
        (r"se descarta \[{}\]",               ["no concluyente de [{}]", "no es confirma [{}]"]),
        (r"es descarta \[{}\]",               ["no concluyente de [{}]", "no es confirma [{}]"]),

        # --- GENEMUTATION / No ---
        (r"no mutacio \[{}\]",                ["negatiu per [{}]", "no es detecta [{}]", "wild type [{}]"]),
        (r"no mutacion \[{}\]",               ["negativo para [{}]", "no se detecta [{}]", "wild type [{}]"]),
        (r"wild type \[{}\]",                 ["no mutacio [{}]", "negatiu per [{}]"]),

        # --- TREATMENT / No ---
        (r"rebutja \[{}\]",                   ["no accepta [{}]", "no vol [{}]", "refusa [{}]"]),
        (r"no vol fer-se \[{}\]",             ["rebutja [{}]", "no accepta [{}]"]),
        (r"STOP \[{}\]",                      ["suspensio de [{}]", "s'atura [{}]", "se suspende [{}]"]),
        (r"\[sense tractament\]",             ["[Sin tto]", "[Sense tto]", "[no tractament]"]),
        (r"\[sin tto\]",                      ["[sense tractament]", "[Sense tto]", "[no tractament]"]),
        (r"\[sense tto\]",                    ["[sense tractament]", "[Sin tto]", "[no tractament]"]),
        (r"\[placebo\]",                      ["[sin tratamiento activo]", "[no tractament]"]),

        # --- SMOKER / No ---
        (r"no (es |es )?(fumadora?|fumador)\]?", ["no fuma", "no es fumador", "no fumadora"]),
        (r"ni es fumador",                    ["no fuma", "no es fumador"]),

        # --- PATRONES ORIGINALES que funcionaban (mantener) ---
        (r"no te \[{}\]",                     ["es descarta [{}]", "negatiu per [{}]"]),
        (r"no tiene \[{}\]",                  ["se descarta [{}]", "negativo para [{}]"]),
        (r"\[{}\] descartado",                ["se excluye [{}]", "no confirma [{}]"]),
        (r"sin \[{}\]",                       ["no muestra [{}]", "niega [{}]"]),
    ]
    
    context_norm      = normalize(context)
    entity_text_norm  = normalize(entity_text)
    entity_escaped    = re.escape(entity_text_norm)
    
    new_contexts = []
    entity_escaped = re.escape(entity_text)
    
    for pat, replacements in patterns:
        full_pat = pat.format(entity_escaped)
        if re.search(full_pat, context_norm, re.IGNORECASE):
            for rep in replacements:
                # ✅ La sustitución se aplica sobre el contexto ORIGINAL
                # para no perder los acentos en el texto generado
                new_context = re.sub(
                    full_pat,
                    lambda m, r=rep.format(entity_text): r[0].upper() + r[1:] if m.group(0)[0].isupper() else r,
                    context_norm,  # trabajamos sobre el normalizado
                    flags=re.IGNORECASE
                )
                new_contexts.append(new_context)
    
    return new_contexts

def prepare_augmented_data(samples):
    augmented = []
    
    # Contadores de debug
    total_negation_samples = 0
    pattern_hit  = 0
    pattern_miss = 0
    total_generated = 0
    pattern_hit_detail = {}  # qu� patrones se disparan m�s

    for sample in samples:
        augmented.append(sample)
        
        if sample["attribute"] in ["Discarded", "No"]:
            total_negation_samples += 1
            variations = augment_negation(sample["context"], sample["entity_text"])
            
            if variations:
                pattern_hit += 1
                total_generated += len(variations)
                for v_context in variations:
                    new_sample = sample.copy()
                    new_sample["context"] = v_context
                    augmented.append(new_sample)
            else:
                pattern_miss += 1
                for _ in range(3):
                    augmented.append(sample.copy())
                    
    # Reporte
    print("=== Debug Augmentacion ===")
    print(f"  Total muestras con No/Discarded:   {total_negation_samples}")
    print(f"  Patrones detectados (variaciones):  {pattern_hit}  ({100*pattern_hit/total_negation_samples:.1f}%)")
    print(f"  Sin patron (solo duplicados x3):    {pattern_miss}  ({100*pattern_miss/total_negation_samples:.1f}%)")
    print(f"  Frases nuevas generadas:            {total_generated}")
    print(f"  Total dataset augmentado:           {len(augmented)}")

    return augmented

# Train Augmentation
train_set = prepare_augmented_data(train_orig)
random.shuffle(train_set)

# Val and Test remain original
val_set  = val_orig
test_set = test_orig

print(f"After augmentation  — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# Print distributions before and after augmentation

print("\n=== Distribution TRAIN (augmented) ===")
attr_counts = Counter(r["attribute"] for r in train_set)
total = sum(attr_counts.values())
for attr, count in sorted(attr_counts.items(), key=lambda x: -x[1]):
    print(f"  {attr:<15} {count:>5}  ({100*count/total:.1f}%)")

print("\n=== Distribution TEST (original) ===")
attr_counts_test = Counter(r["attribute"] for r in test_set)
total_test = sum(attr_counts_test.values())
for attr, count in sorted(attr_counts_test.items(), key=lambda x: -x[1]):
    print(f"  {attr:<15} {count:>5}  ({100*count/total_test:.1f}%)")


# ## 3. Fine Tune XLM-RoBERTa


# fine tune with qlora
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

def check_dataset_distribution(dataset, name="Validation"):
    # Extraemos las etiquetas del dataset de Hugging Face
    labels = dataset["labels"].tolist() if isinstance(dataset["labels"], torch.Tensor) else dataset["labels"]
    counts = Counter(labels)
    
    print(f"\n=== Distribución en {name} set ===")
    for i in range(len(unique_attributes)):
        label_name = id2label[i]
        count = counts.get(i, 0)
        print(f"  ID {i} ({label_name:<15}): {count} muestras")
    
    if len(counts) < len(unique_attributes):
        missing = [id2label[i] for i in range(len(unique_attributes)) if i not in counts]
        print(f"\n⚠️ ALERTA: Faltan las siguientes clases en {name}: {missing}")

# Ejecutar la comprobación
check_dataset_distribution(ds_val, "Validation")
check_dataset_distribution(ds_train, "Train")


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
    
    report = classification_report(
        labels, predictions,
        target_names=unique_attributes,
        output_dict=True,
        zero_division=0
    )
    
    print("\n--- F1 por clase ---")
    for cls in unique_attributes:
        if cls in report:
            f1 = report[cls]["f1-score"]
            support = report[cls]["support"]
            print(f"  {cls:<15} F1={f1:.3f}  (n={support})")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

attr_counts = Counter(s["attribute"] for s in train_set)
total = sum(attr_counts.values())

class_weights = torch.tensor([
    total / (len(unique_attributes) * attr_counts.get(id2label[i], 1))
    for i in range(len(unique_attributes))
], dtype=torch.float)

print("=== Class weights ===")
for i, w in enumerate(class_weights):
    print(f"  {id2label[i]:<15} {w:.3f}  (n={attr_counts.get(id2label[i], 0)})")

training_args = TrainingArguments(
    output_dir="cr-multia/attribute_association/FINETUNNING/RESULTS_BERT_v3",
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

def focal_loss(logits, labels, alpha=None, gamma=2.0):
    ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')(logits, labels)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = focal_loss(outputs.logits, labels, alpha=self.class_weights)
        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
)

trainer.train()

# save adapter
OUT_DIR_LLM = "cr-multia/attribute_association/FINETUNNING/RESULTS_BERT_v3"
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
output_path = Path("cr-multia/attribute_association/FINETUNNING/RESULTS_BERT_v3")
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / "bert_results_v3.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

with open(output_path / "attr_report_BERT_FINETUNNED_v3.txt", "w") as f:
    f.write(report)

def save_confusion_matrix(cm_df, output_path="confusion_matrix_roberta_v3.png"):
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

save_confusion_matrix(cm, "confusion_matrix_roberta_v3.png")

