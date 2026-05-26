#!/usr/bin/env python
# coding: utf-8

# # Fine tunning LLMs: LLAMA3

# dependencies
import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import re
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
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

random.seed(43)
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
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# prompt
def format_attribute_prompt(example):
    # Lógica de opciones según el label
    opts_map = {
        "Diagnosis": "[Confirmed, Control, Progression, Suspicion, Discarded]",
        "Smoker": "[Yes, Previous, No]",
        "GeneMutation": "[Yes, No]",
        "Treatment": "[Yes, No]",
        "Exitus": "[Yes, No]",
        "FamilyHistory": "[Yes, No]"
    }
    options = opts_map.get(example['entity_label'], "[Yes, No]")

    system_prompt = f"""Eres un experto en codificacion medica. Tu objetivo es clasificar el estado de la entidad entre corchetes [...] dentro del contexto clinico.

REGLAS ESTRICTAS:
1. Responde UNICAMENTE con un objeto JSON.
2. La clave debe ser "attribute" y el valor una de estas opciones: [{options}].
3. No des explicaciones ni texto adicional.
"""
    
    user_prompt = f"""Analiza el atributo de la siguiente nota clinica segun el contexto:

Contexto:
{example['context']}

Entidad: "{example['entity_text']}"

Label de la entidad: {example['entity_label']}

Atributo:"""
   
    response_json = json.dumps({"attribute": example['attribute']})
    
    full_text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response_json}<|eot_id|>"
    )
    
    return {"text": full_text}

ds_train = Dataset.from_list(train_set).map(format_attribute_prompt)
ds_val = Dataset.from_list(val_set).map(format_attribute_prompt)

columns_to_remove = ["context", "entity_text", "entity_label", "attribute"]

ds_train = ds_train.remove_columns(columns_to_remove)
ds_val = ds_val.remove_columns(columns_to_remove)

# check
print(ds_train[0].keys())

# model config

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
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
    output_dir="./llama3-attribute-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds_train, 
    eval_dataset=ds_val,
    args=sft_config,
)

trainer.train()


# save adapter
OUT_DIR_LLM = "/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/FINETUNNING/RESULTS_LLAMA_confmatrix/"
out = trainer.save_model(OUT_DIR_LLM)
out
print("Adapter would be saved to:", OUT_DIR_LLM)


# ## 4. Get Results

def evaluate_attribute_model(model, tokenizer, test_data):
    model.eval()
    y_true = []
    y_pred = []
    results_log = []

    print(f"Evaluando {len(test_data)} muestras de test...")

    for example in tqdm(test_data):
        # same format than train
        opts_map = {
            "Diagnosis": "[Confirmed, Control, Progression, Suspicion, Discarded]",
            "Smoker": "[Yes, Previous, No]",
            "GeneMutation": "[Yes, No]",
            "Treatment": "[Yes, No]",
            "Exitus": "[Yes, No]",
            "FamilyHistory": "[Yes, No]"
        }
        options = opts_map.get(example['entity_label'], "[Yes, No]")

        system_prompt = f"""Eres un experto en codificacion medica. Tu objetivo es clasificar el estado de la entidad entre corchetes [...] dentro del contexto clinico.

REGLAS ESTRICTAS:
1. Responde UNICAMENTE con un objeto JSON.
2. La clave debe ser "attribute" y el valor una de estas opciones: [{options}].
3. No des explicaciones ni texto adicional.
"""

        user_prompt = f"""Analiza el atributo de la siguiente nota clinica segun el contexto:

        Contexto:
        {example['context']}

        Entidad: "{example['entity_text']}"

        Label de la entidad: {example['entity_label']}

        Atributo:"""

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                temperature=0.1, 
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        prediction = "Parsing_Error"
        try:
            # search pattern {"attribute": "value"}
            match = re.search(r'\"attribute\":\s*\"(\w+)\"', generated_text)
            if match:
                prediction = match.group(1)
            else:
                # Fallback: si no es JSON, buscamos si alguna opción válida está en el texto
                valid_options = options.replace("[", "").replace("]", "").split(", ")
                for opt in valid_options:
                    if opt.lower() in generated_text.lower():
                        prediction = opt
                        break
        except:
            pass

        y_true.append(example['attribute'])
        y_pred.append(prediction)
        
        results_log.append({
            "context": example['context'],
            "entity_label": example["entity_label"],
            "true": example['attribute'],
            "pred": prediction,
            "raw": generated_text
        })

    print("\nAttribute Classification Report")
    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    
    return report, results_log

report, results = evaluate_attribute_model(model, tokenizer, test_set)

# save results
output_path = Path("/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/FINETUNNING/RESULTS_LLAMA_confmatrix")
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / "llama3_results_v1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

with open(output_path / "attr_report_LLAMA3_FINETUNNED_v1.txt", "w") as f:
    f.write(report)

# Reconstruct true and predicted labels from results
all_true = [r["true"] for r in results]
all_pred = [r["pred"] for r in results]

# Gather all unique labels present (preserving a logical order)
all_classes = sorted(set(all_true + all_pred))

# Build and save confusion matrix
cm = confusion_matrix(all_true, all_pred, labels=all_classes)
cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)

cm_path = output_path / "confusion_matrix_llama3_attr_v1.csv"
cm_df.to_csv(cm_path)
print(f"Confusion matrix saved to {cm_path}")

