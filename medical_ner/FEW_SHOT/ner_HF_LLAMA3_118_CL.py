#!/usr/bin/env python
# coding: utf-8

# # Epic 2 HF: First Pipeline Generic LLM

# In[1]:


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
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoProcessor
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


# ## Authentification HF

# In[ ]:


login(token="YOUR_HF_TOKEN_HERE")


# ## Data Load

# In[5]:


file_path = 'PATH_TO_YOUR_DATA'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# first note and annotations
first_note = data[0]
print(f"Text: {first_note['data']['comment']}")
print(f"Annotations: {first_note['annotations'][0]['result']}")


# ## Functions Definition

# In[6]:


# pydantic model
ALLOWED_LABELS = [
    "Diagnosis", "Treatment", "Response", "Date", "Blasts", 
    "GeneMutation", "VAF", "ProteinChange", "cDNAChange", 
    "Origin", "Karyotype", "Risk", "FamilyHistory", "Smoker", "Exitus"
]

class MedicalEntity(BaseModel):
    text: str = Field(description="Raw Text Clinical Note")
    label: str 

    @model_validator(mode='before')
    @classmethod
    def handle_hallucinations(cls, data):
        if isinstance(data, dict):
            # if label is not in ALLOWED_LABELS
            if data.get("label") not in ALLOWED_LABELS:
                # change to unknown
                data["label"] = "Unknown"
            
        return data

class NERResponse(BaseModel):
    entities: List[MedicalEntity]

# In[7]:


system_prompt = """Eres un experto en extracción de información médica (NER). 
Tu objetivo es identificar entidades clínicas específicas en notas de texto y extraer sus atributos.

REGLAS DE EXTRACCIÓN:
1. Extrae únicamente las entidades que correspondan estrictamente a las siguientes categorias:
   [ "Diagnosis", "Treatment", "Response", "Date", "Blasts", "GeneMutation", "VAF", "ProteinChange", "cDNAChange", "Origin", "Karyotype", "Risk", "FamilyHistory", "Smoker", "Exitus" ]

    A continuación una explicación de cada categoría:
    - Diagnosis: Enfermedades o sospechas (ej: SMD, Leucemia, Anemia).
    - Treatment: Fármacos o terapias (ej: AZA, LENA, Quimioterapia, Trasplante).
    - Response: Respuesta al tratamiento (ej: Remisión completa, Estable, Progresión).
    - Date: Fechas de eventos clínicos.
    - Blasts: Porcentaje de blastos en médula o sangre (ej: 5% blastos).
    - GeneMutation: Nombre del gen mutado (ej: JAK2, TP53, ASXL1).
    - VAF: Frecuencia alélica de la variante (ej: VAF 40%, AF 0.2).
    - ProteinChange: Cambio en la proteína (ej: p.V617F, p.R172H).
    - cDNAChange: Cambio en la secuencia de nucleótidos (ej: c.1849G>T).
    - Origin: Origen de la muestra (ej: Variante Germinal o Somatica).
    - Karyotype: Resultados del cariotipo o alteraciones específicas (ej: 46,XX, del(5q)).
    - Risk: Escalas de riesgo (ej: IPSS-R muy alto, Riesgo intermedio).
    - FamilyHistory: Antecedentes médicos familiares.
    - Smoker: Estado tabáquico del paciente.
    - Exitus: Mención de fallecimiento.

2. El output debe ser estrictamente JSON siguiendo el esquema Pydantic.
3. Si una entidad no tiene un status claro, deja el campo como null.
4. No inventes etiquetas. No des explicaciones.

Aquí te dejo algunos ejemplos:

- Ejemplo 1:
    Texto: "DX amb 75 anys agost 2011 de CRMD IPSS-R 2 (low), no tractament. Progressió AREB-1 el 15/06/2015, es tracta amb EPO (no respon) i amb acid valproic."
    Respuesta: {"entities": [{'text': 'agost 2011', 'label': 'Date'}, {'text': 'CRMD', 'label': 'Diagnosis'}, {'text': 'IPSS-R 2 (low)', 'label': 'Risk'}, {'text': 'no tractament', 'label': 'Treatment'}, {'text': 'AREB-1', 'label': 'Diagnosis'}, {'text': '15/06/2015', 'label': 'Date'}, {'text': 'EPO', 'label': 'Treatment'}, {'text': 'acid valproic', 'label': 'Treatment'}, {'text': 'no respon', 'label': 'Response'}]}

- Ejemplo 2:
    Texto: "L'aspirat de MO fet al novembre del 2021 no era concloent de SMD, ja que la displàsia era inespecífica i compatible an MO reactiu a la seva malaltia de base (tumor sòlid, que estava en progressió en aquell moment), i no hi havia marcador clonal."
    Respuesta: {"entities": [{'text': 'SMD', 'label': 'Diagnosis'}, {'text': 'novembre del 2021', 'label': 'Date'}, {'text': 'displàsia', 'label': 'Diagnosis'}, {'text': 'tumor sòlid', 'label': 'Diagnosis'}]}

- Ejemplo 3:
    Texto: "Sospita SMD"
    Respuesta: {"entities": [{'text': 'SMD', 'label': 'Diagnosis'}]}

- Ejemplo 4:
    Texto: "Anemia macrocítica."
    Respuesta: {"entities": [{'text': 'Anemia macrocítica', 'label': 'Diagnosis'}]}

- Ejemplo 5:
    Texto: "Posible germinal. Hay muestra de pelo HF_2149"
    Respuesta: {"entities": [{'text': 'germinal', 'label': 'Origin'}]}

"""


# In[8]:


def process_ner(text):
    # Preparem el prompt amb instrucció de format
    prompt_format = """{
    "entities": [
        {"text": "texto exacto", "label": "Categoría", "status": null}   ]
    }"""

    user_prompt = f"""Analiza la siguiente nota clínica:
    {text}

    REGLA ESTRICTA: 
    1. Extrae el texto EXACTO.
    2. Responde ÚNICAMENTE con el objeto JSON final. 
    3. No incluyas explicaciones ni el esquema de validación.
    
    Formato esperado:
    {prompt_format}"""

    # Format de chat (molt important per a Command R i Llama 3)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Generació
    outputs = pipe(
        messages, 
        max_new_tokens=1024, 
        do_sample=False, 
        temperature=0.0, # deterministic
        eos_token_id=terminators, # Added terminators here
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"DEBUG: Content: {outputs[0]['generated_text']}")
    
    # Extreure el text de la resposta
    raw_content = outputs[0]["generated_text"]
    
    # Netegem possibles Markdown blocks (```json ... ```) si el model els posa
    clean_json = raw_content.replace("```json", "").replace("```", "").strip()
    
    # Validem i convertim a objecte Pydantic
    return NERResponse.model_validate_json(clean_json)


# In[8]:


def get_evaluation_lists(all_notes_data, all_predictions):
    y_true = []
    y_pred = []
    
    # create dictionary with ids
    # id to string
    preds_by_id = {str(p['id']): p['prediction'] for p in all_predictions}
    
    for entry in all_notes_data:
        entry_id = str(entry.get('id'))
        
        # look at ids
        if entry_id not in preds_by_id:
            print(f"Note {entry_id} not included: LLM did not predict labels.")
            continue
            
        try:
            # ground truth extraction
            gt_results = entry.get('annotations', [{}])[0].get('result', [])
            gt_map = {
                r['value']['text']: r['value']['labels'][0] 
                for r in gt_results if 'labels' in r.get('value', {})
            }
            
            # extract prediction
            pred_data = preds_by_id[entry_id]
            pred_entities = pred_data.get('entities', [])
            pred_map = {
                e['text']: e['label'] for e in pred_entities
            }
            
            # align with ids
            all_mentions = set(gt_map.keys()) | set(pred_map.keys())
            
            for mention in all_mentions:
                y_true.append(gt_map.get(mention, "O"))
                y_pred.append(pred_map.get(mention, "O"))
                
        except Exception as e:
            print(f"❌ Error processing note {entry_id}: {e}")
            continue 
            
    return y_true, y_pred


# ## Llama3 8B Model HF

# In[4]:


# hugging face model
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


# In[9]:


results = []

print(f"Initializing process for {len(data)} clinical notes...\n")

for i, entry in enumerate(data, 1):
    data_obj = entry.get('data', {})
    text_to_analyze = data_obj.get('comment') or entry.get('comment')

    if not text_to_analyze:
        continue
        
    try:
        prediction = process_ner(text_to_analyze)
        
        result_entry = {
            "id": entry.get('id'),
            "raw_text": text_to_analyze,  
            "prediction": prediction.dict()
        }
        
        results.append(result_entry)
        
        print(f"[{i}/{len(data)}] ID {entry.get('id')}: {len(prediction.entities)} entidades.")
        
    except Exception as e:
        print(f"[{i}/{len(data)}] Error en nota ID {entry.get('id')}: {e}")

print(f"\nProcess finished")


# In[75]:


results[0]


# In[78]:


y_true, y_pred = get_evaluation_lists(data, results)

labels = [l for l in set(y_true) if l != "O"] 
report = classification_report(y_true, y_pred, labels=labels)

print("Medical NER Classification Report")
print(report)


# In[85]:


with open("cr-multia/medical_ner/FEW_SHOT/llama_results_118.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
    
print("Results saved to llama_results_118.json")


# In[ ]:


with open("cr-multia/medical_ner/FEW_SHOT/medical_ner_report_LLAMA3_118.txt", "w") as f:
    f.write("Medical NER Classification Report\n")
    f.write(report)

print("Report saved to medical_ner_report_LLAMA3_118.txt")

