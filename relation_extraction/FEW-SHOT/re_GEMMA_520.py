# FEW-SHOT BENCHMARK AGENT 3

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

system_prompt= """Eres un experto en extracción de relaciones entre entidades mèdicas. Tu objetivo es identificar conexiones entre entidades médicas y clasificarlas basándote en identificadores únicos (IDs). Estos identificadores se encuentran junto a las entidades entre parentesis.

INPUT FORMAT:
Recibirás un texto donde las entidades están marcadas como [texto](ID).

REGLAS DE RELACIONES (ESTRICTAS): 
Clasifica las relaciones usando alguna de las etiquetas que salen en el listado de reglas.
1. [treats]: Conecta Treatment -> Diagnosis. Nota: Aunque el atributo del tratamiento sea "NO", debe relacionarse al diagnóstico si existe el vínculo clínico.
2. [occurred_on]: Conecta cualquier Etiqueta -> Date. Se usa para vincular entidades con sus fechas correspondientes. Si hay dos fechas (inicio y fin), crea dos relaciones distintas.
3. [has_value]: 
   - Diagnosis -> Risk (para especificar el estado de riesgo).
   - Diagnosis -> Blasts (vínculo directo con el % de blastos).
   - Mutation -> VAF (el % de VAF se relaciona directamente con la mutación).
4. [has_outcome]: Treatment -> Response. Identifica las respuestas que el paciente tiene a un tratamiento específico.
5. [associated_with]:
   - Diagnosis/Mutation -> Origin (vínculo con el origen de la enfermedad: somático o germinal).
   - Diagnosis -> Karyotype (vínculo con el cariotipo asociado al diagnóstico).
   - Diagnosis -> Mutation (vínculo directo cuando una mutación define o se asocia al diagnóstico).
6. [caused_by]: 
   - Exitus -> Diagnosis/Response (cuando la causa de muerte es un diagnóstico o una respuesta).
   - Diagnosis(TRMN-like) -> Treatment(Atributo NO) (vínculo obligatorio si el diagnóstico es tipo TRMN-like).
7. [has_dose]: Treatment -> Treatment (cuando una etiqueta es el nombre del fármaco y la otra es la dosis o ciclos).
8. [has_change_in]: Mutation -> cDNAChange/ProteinChange (vínculo de cambios moleculares específicos).



REGLAS PARA EL FORMATO DE SALIDA:
- Responde EXCLUSIVAMENTE con un JSON array.
- No añadidas explicaciones ni texto introductorio.
- Formato: [{"from_id": "ID", "to_id": "ID", "type": "relacion"}]

EJEMPLOS:
Ejemplo 1:
- Texto: "Inicia [LENA](GjHyzeOUy9) en [2008](hdPmyf2U27)"
- Entidades: [LENA](GjHyzeOUy9): Treatment, [2008](hdPmyf2U27): Date
- Salida: [{"from_id": "GjHyzeOUy9", "to_id": "hdPmyf2U27", "type": "occurred_on"}]

Ejemplo 2:
- Texto: "Es [exitus](BjaaaQWEy4) en [diciembre de 2020](GjaaaeO123) por [sepsis](GjaaaeOUy9)"
- Entidades: [exitus](BjaaaQWEy4): Exitus, [diciembre de 2020](GjaaaeO123): Date, [sepsis](GjaaaeOUy9): Diagnosis
- Salida: [
    {"from_id": "BjaaaQWEy4", "to_id": "GjaaaeOUy9", "type": "caused_by"}, 
    {"from_id": "BjaaaQWEy4", "to_id": "GjaaaeO123", "type": "occurred_on"}
  ]

Ejemplo 3:
- Texto: "DX: [SMD MLD](Gjfwe1eOUy9). [Low risk](FT65EFdjwh). Panell: [TET2](23yfhfjkRPL) ([VAF 12%](gsf7395DFth))"
- Entidades: [SMD MLD](Gjfwe1eOUy9): Diagnosis, [Low risk](FT65EFdjwh): Risk, [TET2](23yfhfjkRPL): Mutation, [VAF 12%](gsf7395DFth): VAF
- Salida: [
    {"from_id": "Gjfwe1eOUy9", "to_id": "FT65EFdjwh", "type": "has_value"}, 
    {"from_id": "23yfhfjkRPL", "to_id": "gsf7395DFth", "type": "has_value"}
  ]

Ejemplo 4:
- Texto: "En [2014](ertfsd34) se diagnostica [CRDM](8OPdt34)"
- Entidades: [2014](ertfsd34): Date, [CRDM](8OPdt34): Diagnosis
- Salida: [{"from_id": "8OPdt34", "to_id": "ertfsd34", "type": "occurred_on"}]
(Nota: Aunque la fecha ertfsd34  aparece antes, la regla obliga a que la entidad medica 8OPdt34 sea from_id y la fecha sea to_id).
"""

# fine tune model
model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    low_cpu_mem_usage=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>")
]


# Process the tasks

results = []

print(f"Initializing inference for {len(all_tasks)} tasks...")

for task in tqdm_cli(all_tasks):

    indexed_text = format_text_with_ids(task)

    entity_legend = "\n".join([f"- {e['id']}: {e['text']} ({e['label']})" for e in task["entities"]])

    user_prompt = f"""Analiza la siguiente frase y sus entidades para extraer relaciones entre dichas entidades:

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
output_file = "re_results_gemma_520_v1.json"
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
            gt_set = set([
                (r.get('from_id', ''), r.get('to_id', ''))
                for r in item["ground_truth"]
                if isinstance(r, dict) and r.get('type') == rel_type])
            pred_set = set([
                (r.get('from_id', ''), r.get('to_id', ''))
                for r in item["prediction"]
                if isinstance(r, dict) and r.get('type') == rel_type])

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

with open("cr-multia/relation_extraction/FEW_SHOT/report_GEMMA_520_v1.txt", "w") as f:
    f.write("Relation Extraction Report\n")
    f.write("=" * 70 + "\n")
    f.write(report.to_string(index=False))

print("Report saved to report_GEMMA_520_v1.txt")

