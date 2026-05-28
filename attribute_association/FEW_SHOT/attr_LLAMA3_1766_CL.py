# FEW-SHOT BENCHMARK AGENT 2

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

def prepare_ground_truth_tasks(dataset):
    tasks = []
    
    for item in dataset:
        text = item["data"]["comment"]
        annotations = item.get("annotations", [{}])[0].get("result", [])
        
        # We need to link 'labels' with their corresponding 'choices'
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

        # Now we merge them into a single task list using the full text
        for region_id, label_info in id_to_label.items():
            start, end = label_info["start"], label_info["end"]
            
            # Construct the full text with the specific entity highlighted in brackets
            full_context = text[:start] + f"[{label_info['text']}]" + text[end:]
            
            tasks.append({
                "entity_text": label_info["text"],
                "entity_label": label_info["label"],
                "true_attribute": id_to_choice.get(region_id, "None"), 
                "context": full_context
            })
            
    return tasks

# Prepare the data
all_tasks = prepare_ground_truth_tasks(ner_dataset)
print(f"Extracted {len(all_tasks)} entities with their attributes.")

# Example of the first task
print(json.dumps(all_tasks[10], indent=2))


# ## Functions Definition

def generate_llama_prompt(task):
    # Mapping logic for the prompt instructions
    if task['entity_label'] == "Diagnosis":
        options = "[Confirmed, Control, Progression, Suspicion, Discarded]"
    elif task['entity_label'] == "Smoker":
        options = "[Yes, Previous, No]"
    elif task['entity_label'] == "GeneMutation":
        options = "[Yes, No]"
    elif task['entity_label'] == "Treatment":
        options = "[Yes, No]"
    elif task['entity_label'] == "Exitus":
        options = "[Yes, No]"
    elif task['entity_label'] == "FamilyHistory":
        options = "[Yes, No]"
    else:
        return None

    system_prompt= f"""

Eres un experto en codificacion medica. Tu objetivo es clasificar el estado de la entidad entre corchetes [...] dentro del contexto clinico.

REGLAS ESTRICTAS:
1. Responde UNICAMENTE con un objeto JSON.
2. La clave debe ser "attribute" y el valor una de estas opciones: [{options}].
3. No des explicaciones ni texto adicional.

EJEMPLOS:
- Contexto: "Es [exfumador] de fa 6 anys". Entidad: "exfumador". Label de la entidad: "Smoker" -> {{"attribute": "Previous"}}
- Contexto: "Progresa a [LMA]". Entidad: "LMA". Label de la entidad: "Diagnosis" -> {{"attribute": "Progression"}}
- Contexto: "Sospita [SMD]". Entidad: "SMD". Label de la entidad: "Diagnosis" -> {{"attribute": "Suspicion"}}
- Contexto: "SMD desde 2020. [Sin tto]". Entidad: "Sin tto". Label de la entidad: "Treatment" -> {{"attribute": "No"}}
- Contexto: "[No tiene antecedentes familiares]". Entidad: "No tiene antecedentes familiares". Label de la entidad: "FamilyHistory" -> {{"attribute": "No"}}
- Contexto: "SMD desde 2008 con del 5q. [Padre con Ca de colon]". Entidad: "Padre con Ca de colon". Label de la entidad: "FamilyHistory" -> {{"attribute": "Yes"}}

    """

    user_prompt = f"""Analiza el atributo de la siguiente nota clinica segun el contexto:

Contexto:
{task['context']}

Entidad: "{task['entity_text']}" 
Label de la entidad: {task['entity_label']}

Atributo:"""
    return user_prompt, system_prompt


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

# Filtering tasks that have a valid prompt (ignoring Date, Blasts, etc.)
ALLOWED_LABELS = ["Diagnosis", "Smoker", "GeneMutation", "Treatment", "Exitus", "FamilyHistory"]

valid_tasks = [t for t in all_tasks if t['entity_label'] in ALLOWED_LABELS]

print(f"Starting inference for {len(valid_tasks)} attribute classification tasks...")

def clean_prediction(raw_output, allowed_options):
    raw_output = raw_output.lower()
    
    for option in allowed_options:
        if option.lower() in raw_output:
            return option
    return "Parsing_Error"


for task in tqdm_cli(valid_tasks):
    user_prompt, system_prompt = generate_llama_prompt(task)
    
    if user_prompt is None:
        continue

    label = task['entity_label']
    if label == "Diagnosis":
        current_options = ["Confirmed", "Control", "Progression", "Suspicion", "Discarded"]
    elif label == "Smoker":
        current_options = ["Yes", "Previous", "No"]
    elif label in ["GeneMutation", "Treatment", "Exitus", "FamilyHistory"]:
        current_options = ["Yes", "No"]
    else:
        continue

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    outputs = pipe(
        messages,
        max_new_tokens=20,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False 
    )
    
    raw_response = outputs[0]["generated_text"].strip()
    
    prediction = clean_prediction(raw_response, current_options)

    results.append({
        "entity_text": task["entity_text"],
        "entity_label": task["entity_label"],
        "true_attribute": task["true_attribute"],
        "pred_attribute": prediction,
        "raw_llm_out": raw_response
    })
    


with open("cr-multia/attribute_association/FEW_SHOT/llama3_results_1766.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
    
print("Results saved to llama3_results_1766.json")


# Eval results

y_true = []
y_pred = []

for item in results:
    t = item["true_attribute"]
    p = item["pred_attribute"]
    
    # Optional: Filter out "None" if you only want to see 
    # the performance on cases where a choice was actually made
    if t != "None" and p != "Parsing_Error":
        y_true.append(t)
        y_pred.append(p)

# Generate the Report
print("Attribute Classification Report")
print("-" * 60)
report = classification_report(y_true, y_pred, zero_division=0)
print(report)

# Breakdown by Entity Type
print("\nAccuracy by Entity Type:")
entity_types = set([item["entity_label"] for item in results])

for etype in entity_types:
    etype_true = [item["true_attribute"] for item in results if item["entity_label"] == etype and item["true_attribute"] != "None"]
    etype_pred = [item["pred_attribute"] for item in results if item["entity_label"] == etype and item["true_attribute"] != "None"]
    
    if len(etype_true) > 0:
        correct = sum(1 for tr, pr in zip(etype_true, etype_pred) if tr == pr)
        print(f"{etype:15}: {correct/len(etype_true):.2%} (Support: {len(etype_true)})")


with open("cr-multia/attribute_association/FEW_SHOT/report_LLAMA3_1766.txt", "w") as f:
    f.write("Attribute Classification Report\n")
    f.write(report)

print("Report saved to report_LLAMA3_1766.txt")

