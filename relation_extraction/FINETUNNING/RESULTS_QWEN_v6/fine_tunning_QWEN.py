#!/usr/bin/env python
# coding: utf-8

# # Fine tunning LLMs: QWEN

# dependencies
import os
import json
import pandas as pd
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import re
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import DataCollatorForCompletionOnlyLM
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig, 
    Trainer, 
    DataCollatorForTokenClassification
)
import seqeval
from huggingface_hub import login


# login to hugging face
login(token="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")


# ## 1. Data Load and Preprocessing

dataset_path = Path("/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/data/subsample_2566_FIXED.json")

with open(dataset_path, "r", encoding="utf-8") as f:
    ner_dataset = json.load(f)

print(f"Loaded {len(ner_dataset)} clinical notes")


ner_dataset[0]


# prepare dataset

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
                    "label": r["value"]["labels"][0],
                    "start": r["value"]["start"],   # ✅ real offset
                    "end": r["value"]["end"]         # ✅ real offset
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


# add ids in the text so that the model knows how to refer to each entity
def format_text_with_ids(task):
    text = task["text"]

    entities = sorted(task["entities"], key=lambda x: x["start"], reverse=True)
    
    indexed_text = text
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        # consistency debug
        assert indexed_text[start:end] == ent["text"] or True, \
            f"Offset mismatch: '{indexed_text[start:end]}' != '{ent['text']}'"
        indexed_text = indexed_text[:start] + f"[{ent['text']}]({ent['id']})" + indexed_text[end:]
    
    return indexed_text

all_tasks = prepare_relation_tasks(ner_dataset)

# train test val split 

random.seed(43)
ner_dataset_shuffled = all_tasks.copy()
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

def debug_relation_counts(datasets_dict):
    """
    datasets_dict: {'Train': train_set, 'Val': val_set, 'Test': test_set}
    """
    all_stats = []
    
    for name, dataset in datasets_dict.items():
        counts = {}
        total_rels = 0
        for task in dataset:
            for rel in task['relations']:
                rel_type = rel['type']
                counts[rel_type] = counts.get(rel_type, 0) + 1
                total_rels += 1
        
        counts['TOTAL_RELATIONS'] = total_rels
        counts['Total_Notes'] = len(dataset)
        all_stats.append(pd.Series(counts, name=name))

    df_stats = pd.concat(all_stats, axis=1).fillna(0).astype(int)
    print("\n=== DEBUG: Relations distribution per set ===")
    print(df_stats)
    return df_stats

stats = debug_relation_counts({
    'Train': train_set, 
    'Val': val_set, 
    'Test': test_set
})

# ## 3. Fine Tune QWen2.5 

# fine tune with qlora
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# prompt
def format_re_prompt(task):

    indexed_text = format_text_with_ids(task)
    
    entity_legend = "\n".join([f"- {e['id']}: {e['text']} ({e['label']})" for e in task["entities"]])
    
    system_prompt= """Eres un experto en extraccion de relaciones entre entidades medicas. Tu objetivo es identificar conexiones entre dichas entidades medicas y clasificarlas basandote en identificadores unicos (IDs). Estos identificadores se encuentran junto a las entidades entre parentesis.

INPUT FORMAT:
Recibiras un texto donde las entidades estan marcadas de la siguiente manera: [entidad](ID).

REGLAS DE RELACIONES (ESTRICTAS): 
Clasifica las relaciones usando alguna de las etiquetas que salen en el listado de reglas.
1. [treats]: Conecta Treatment -> Diagnosis. Nota: Aunque el atributo del tratamiento sea "NO", debe relacionarse al diagnostico si existe el vinculo clinico.
2. [occurred_on]: Conecta cualquier Etiqueta -> Date. Se usa para vincular entidades con sus fechas correspondientes. Si hay dos fechas (inicio y fin), crea dos relaciones distintas.
3. [has_value]: 
   - Diagnosis -> Risk (para especificar el estado de riesgo).
   - Diagnosis -> Blasts (vinculo directo con el % de blastos).
   - Mutation -> VAF (el % de VAF se relaciona directamente con la mutacion).
4. [has_outcome]: Treatment -> Response. Identifica las respuestas que el paciente tiene a un tratamiento especifico.
5. [associated_with]:
   - Diagnosis/Mutation -> Origin (vinculo con el origen de la enfermedad: somatico o germinal).
   - Diagnosis -> Karyotype (vinculo con el cariotipo asociado al diagnostico).
   - Diagnosis -> Mutation (vinculo directo cuando una mutacion define o se asocia al diagnostico).
6. [caused_by]: 
   - Exitus -> Diagnosis/Response (cuando la causa de muerte es un diagnostico o una respuesta).
   - Diagnosis(TRMN-like) -> Treatment(Atributo NO) (vinculo obligatorio si el diagnostico es tipo TRMN-like).
7. [has_dose]: Treatment -> Treatment (cuando una etiqueta es el nombre del farmaco y la otra es la dosis o ciclos).
8. [has_change_in]: Mutation -> cDNAChange/ProteinChange (vinculo de cambios moleculares especificos).

REGLAS PARA EL FORMATO DE SALIDA:
- Responde EXCLUSIVAMENTE con un JSON array.
- No incluyas explicaciones ni texto introductorio.
- Formato: [{"from_id": "ID", "to_id": "ID", "type": "relacion"}]
"""

    user_prompt = f"""Analiza el siguiente texto y sus entidades para extraer relaciones entre dichas entidades:

Texto:"{indexed_text}"

Entidades: {entity_legend}

Salida:"""

    response_json = json.dumps(task["relations"], ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response_json}
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    return {"text": full_text}

def format_re_prompt_inference(task):
    indexed_text = format_text_with_ids(task)
    entity_legend = "\n".join([f"- {e['id']}: {e['text']} ({e['label']})" for e in task["entities"]])

    system_prompt= """Eres un experto en extraccion de relaciones entre entidades medicas. Tu objetivo es identificar conexiones entre dichas entidades medicas y clasificarlas basando$

INPUT FORMAT:
Recibiras un texto donde las entidades estan marcadas de la siguiente manera: [entidad](ID).

REGLAS DE RELACIONES (ESTRICTAS):
Clasifica las relaciones usando alguna de las etiquetas que salen en el listado de reglas.
1. [treats]: Conecta Treatment -> Diagnosis. Nota: Aunque el atributo del tratamiento sea "NO", debe relacionarse al diagnostico si existe el vinculo clinico.
2. [occurred_on]: Conecta cualquier Etiqueta -> Date. Se usa para vincular entidades con sus fechas correspondientes. Si hay dos fechas (inicio y fin), crea dos relaciones distintas.
3. [has_value]:
   - Diagnosis -> Risk (para especificar el estado de riesgo).
   - Diagnosis -> Blasts (vinculo directo con el % de blastos).
   - Mutation -> VAF (el % de VAF se relaciona directamente con la mutacion).
4. [has_outcome]: Treatment -> Response. Identifica las respuestas que el paciente tiene a un tratamiento especifico.
5. [associated_with]:
   - Diagnosis/Mutation -> Origin (vinculo con el origen de la enfermedad: somatico o germinal).
   - Diagnosis -> Karyotype (vinculo con el cariotipo asociado al diagnostico).
   - Diagnosis -> Mutation (vinculo directo cuando una mutacion define o se asocia al diagnostico).
6. [caused_by]:
   - Exitus -> Diagnosis/Response (cuando la causa de muerte es un diagnostico o una respuesta).
   - Diagnosis(TRMN-like) -> Treatment(Atributo NO) (vinculo obligatorio si el diagnostico es tipo TRMN-like).
7. [has_dose]: Treatment -> Treatment (cuando una etiqueta es el nombre del farmaco y la otra es la dosis o ciclos).
8. [has_change_in]: Mutation -> cDNAChange/ProteinChange (vinculo de cambios moleculares especificos).

REGLAS PARA EL FORMATO DE SALIDA:
- Responde EXCLUSIVAMENTE con un JSON array.
- No incluyas explicaciones ni texto introductorio.
- Formato: [{"from_id": "ID", "to_id": "ID", "type": "relacion"}]
"""


    user_prompt = f"""Analiza el siguiente texto y sus entidades para extraer relaciones entre dichas entidades:

Texto:"{indexed_text}"

Entidades: {entity_legend}

Salida:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  
    )

    return {
        "prompt": prompt,
        "ground_truth": task["relations"],   
        "text": task["text"]
    }

# use formating for train and validation sets
ds_train = Dataset.from_list(train_set).map(format_re_prompt)
ds_val = Dataset.from_list(val_set).map(format_re_prompt)
ds_test  = Dataset.from_list(test_set).map(format_re_prompt_inference)

columns_to_remove = ["text", "entities", "relations", "indexed_text", "entity_legend"]

ds_train = ds_train.remove_columns([col for col in ds_train.column_names if col != "text"])
ds_val = ds_val.remove_columns([col for col in ds_val.column_names if col != "text"])

ds_test  = ds_test.remove_columns([col for col in ds_test.column_names 
                                    if col not in ["prompt", "ground_truth", "text"]])

# debug check
print(ds_train[0].keys())

# model config
lora_config = LoraConfig(
    r=64, 
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir="/ijc/LABS/SOLE/DATA/tfm_CLG/relation_extraction/FINETUNNING/RESULTS_QWEN_v6",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    bf16=True,
    packing=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataset_text_field="text",
    max_seq_length=1280,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds_train, 
    eval_dataset=ds_val,
    tokenizer=tokenizer, 
    args=sft_config,
)

trainer.train()


# save adapter
OUT_DIR_LLM = "/ijc/LABS/SOLE/DATA/tfm_CLG/relation_extraction/FINETUNNING/RESULTS_QWEN_v6"
out = trainer.save_model(OUT_DIR_LLM)
out
print("Adapter would be saved to:", OUT_DIR_LLM)


# ## 4. Get Results

def evaluate_re_model(model, tokenizer, ds_test):
    model.eval()
    results_log = []

    print(f"Evaluating {len(ds_test)} test sample texts...")

    for example in tqdm(ds_test):
        prompt = example["prompt"]         
        ground_truth = example["ground_truth"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        try:
            match = re.search(r'\[.*\]', generated_text, re.DOTALL)
            pred_json = json.loads(match.group()) if match else []
        except:
            pred_json = []

        results_log.append({
            "text": example["text"],
            "ground_truth": ground_truth,
            "prediction": pred_json,
            "raw_output": generated_text
        })

    return results_log

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


results = evaluate_re_model(model, tokenizer, ds_test)

report = generate_relation_report_table(results)

print("\n--- FINAL REPORT RELATION EXTRACTION ---")
print(report.to_string(index=False))

# save results
output_path = Path("/ijc/LABS/SOLE/DATA/tfm_CLG/relation_extraction/FINETUNNING/RESULTS_QWEN_v6")
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / "qwen_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

with open(output_path / "report_QWEN_FINETUNNED_v6.txt", "w") as f:
    f.write(report.to_string(index=False))

