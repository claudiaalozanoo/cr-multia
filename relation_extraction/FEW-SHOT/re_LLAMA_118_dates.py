#!/usr/bin/env python
# coding: utf-8

# # Epic 2 HF: First Pipeline Generic LLM


### library dependencies

import pandas as pd
import numpy as np
import html
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
from collections import Counter, defaultdict
import sys
from ollama import Client
from tqdm import tqdm as tqdm_cli
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoProcessor, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from numpy.linalg import norm
import json, re, os, string, random
from typing import List, Tuple, Dict, Literal, Optional
from sklearn.model_selection import train_test_split
from joblib import dump
import torch
import pycrfsuite
from sklearn_crfsuite import metrics as crf_metrics
from pydantic import BaseModel, Field, model_validator
from sklearn.metrics import classification_report
from huggingface_hub import login
from pathlib import Path


# ## Authentification HF

login(token="YOUR_HF_TOKEN_HERE")

# ## Data Load

dataset_path = "PATH_TO_YOUR_DATA"

with open(dataset_path, "r", encoding="utf-8") as f:
    ner_dataset = json.load(f)

def prepare_relation_tasks(ls_dataset):
    all_tasks = []

    for item in ls_dataset:
        text = item["data"]["comment"] 
        annotations = item.get("annotations", [{}])[0].get("result", [])

        entities = []
        relations = []

        for r in annotations:
            if r["type"] == "labels":
                entities.append({
                    "id": r["id"],
                    "text": r["value"]["text"],
                    "label": r["value"]["labels"][0]
                })
            
            elif r["type"] == "relation":
                relations.append({
                    "from_id": r["from_id"],
                    "to_id": r["to_id"],
                    "type": r["labels"][0]
                })

        all_tasks.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })

    return all_tasks

all_tasks = prepare_relation_tasks(ner_dataset)

print(f"Extracted {len(all_tasks)} entities with their attributes.")

# Example of the first task
print(json.dumps(all_tasks[10], indent=2))

# add ids in the text so that the model knows how to refer to each entity

def format_text_with_ids(task):
    text = task["text"]

    entities = sorted(task["entities"], key=lambda x: text.find(x["text"]), reverse=True)
    
    indexed_text = text
    for ent in entities:
        start = text.find(ent["text"])
        if start != -1:
            end = start + len(ent["text"])
            # add id nex to entity
            indexed_text = indexed_text[:start] + f"[{ent['text']}]({ent['id']})" + indexed_text[end:]
    
    return indexed_text


# define prompts

system_prompt= """Eres un experto en extracción de relaciones entre entidades mèdicas. Tu objetivo es identificar conexiones entre fechas y las entidades médicas relacionadas. Relaciona los identificadores de las entidades con los de las fechas, estos se encuentran junto a las entidades entre parentesis.

INPUT FORMAT:
Recibirás un texto donde las entidades están marcadas como [texto](ID).

REGLAS DE RELACIONES (ESTRICTAS): 
1. Identifica y clasifica SOLO las relaciones que sean desde Cualquier etiqueta hacia etiqueta DATE usando la label [occurred_on]. Se usa para vincular entidades con sus fechas correspondientes. 
2. Si hay dos fechas y estas se corresponden al inicio y fin de algún proceso crea dos relaciones distintas. En caso que haya alguna fecha que no esta relacionada con nada, NO SE RELACIONA. 
3. Es importante que te fijes en el resto del texto clínico para identificar dichas relaciones. 

REGLAS PARA EL FORMATO DE SALIDA:
- Responde EXCLUSIVAMENTE con un JSON array.
- No añadidas explicaciones ni texto introductorio.
- Formato: [{"from_id": "ID", "to_id": "ID", "type": "occurred_on"}]

EJEMPLOS:
Ejemplo 1:
- Texto: "Inicia [LENA](GjHyzeOUy9) en [2008](hdPmyf2U27)"
- Entidades: [LENA](GjHyzeOUy9): Treatment, [2008](hdPmyf2U27): Date
- Salida: [{"from_id": "GjHyzeOUy9", "to_id": "hdPmyf2U27", "type": "occurred_on"}]

Ejemplo 2:
- Texto: "Es [exitus](BjaaaQWEy4) en [diciembre de 2020](GjaaaeO123) por [sepsis](GjaaaeOUy9)"
- Entidades: [exitus](BjaaaQWEy4): Exitus, [diciembre de 2020](GjaaaeO123): Date, [sepsis](GjaaaeOUy9): Diagnosis
- Salida: [
    {"from_id": "BjaaaQWEy4", "to_id": "GjaaaeO123", "type": "occurred_on"}
  ]
"""

# fine tune model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Llama 3 doesn't have a default pad token, so we map it to eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# token terminators needed in llama3
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# Process the tasks

results = []

print(f"Initializing inference for {len(all_tasks)} tasks...")

for task in tqdm_cli(all_tasks):

    indexed_text = format_text_with_ids(task)

    entity_legend = "\n".join([f"- {e['id']}: {e['text']} ({e['label']})" for e in task["entities"]])

    user_prompt = f"""Analiza el siguiente texto y sus entidades para extraer relaciones entre dichas entidades y sus fechas:

Texto:"{indexed_text}"

Entidades: {entity_legend}

Salida:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    outputs = pipe(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False, # Determinístico para mayor precisión
        temperature=0.0
    )

    raw_response = outputs[0]["generated_text"].strip()
    
    try:
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if json_match:
            pred_relations = json.loads(json_match.group())
        else:
            pred_relations = []
    except Exception as e:
        print(f"Error parseando JSON en una tarea: {e}")
        pred_relations = []

    results.append({
        "original_text": task["text"],
        "indexed_text": indexed_text,
        "labeled_entities": task["entities"],
        "ground_truth": task["relations"],
        "prediction": pred_relations
    })

# save output
output_file = "re_results_llama3_118_v1_dates.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Proceso finalizado. Resultados guardados en {output_file}")


# Eval results
def generate_relation_report_table(results):
    rel_labels = [
        "caused_by", "treats", "occurred_on", "has_change_in",
        "has_value", "has_outcome", "has_dose", "associated_with"
    ]

    rows = []

    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_supports = []

    for rel_type in rel_labels:
        tp, fp, fn = 0, 0, 0

        for item in results:
            gt_set = set([(r['from_id'], r['to_id']) for r in item["ground_truth"] if r['type'] == rel_type])
            pred_set = set([(r['from_id'], r['to_id']) for r in item["prediction"] if r['type'] == rel_type])

            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_supports.append(support)

        rows.append({
            "Relation Type": rel_type,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support
        })

    total_support = sum(all_supports)
    
    macro_precision = sum(all_precisions) / len(rel_labels)
    macro_recall = sum(all_recalls) / len(rel_labels)
    macro_f1 = sum(all_f1s) / len(rel_labels)

    if total_support > 0:
        weighted_precision = sum(p * s for p, s in zip(all_precisions, all_supports)) / total_support
        weighted_recall = sum(r * s for r, s in zip(all_recalls, all_supports)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(all_f1s, all_supports)) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0

    rows.append({"Relation Type": "-"*15, "Precision": None, "Recall": None, "F1-Score": None, "Support": ""})
    
    rows.append({
        "Relation Type": "macro avg",
        "Precision": macro_precision,
        "Recall": macro_recall,
        "F1-Score": macro_f1,
        "Support": total_support
    })
    
    rows.append({
        "Relation Type": "weighted avg",
        "Precision": weighted_precision,
        "Recall": weighted_recall,
        "F1-Score": weighted_f1,
        "Support": total_support
    })

    df_report = pd.DataFrame(rows)

    cols_to_format = ["Precision", "Recall", "F1-Score"]
    for col in cols_to_format:
        df_report[col] = df_report[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")

    return df_report

# run
report = generate_relation_report_table(results)
print("\nRelation Extaction Report")
print("=" * 70)
print(report.to_string(index=False))

with open("cr-multia/relation_extraction/FEW_SHOT/report_LLAMA3_118_v1_dates.txt", "w") as f:
    f.write("Relation Extraction Report\n")
    f.write("=" * 70 + "\n")
    f.write(report.to_string(index=False))

print("Report saved to report_LLAMA3_118_v1_dates.txt")

