#!/usr/bin/env python
# coding: utf-8

# # Fine tunning LLMs: LLAMA3

# In[1]:


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
import pandas as pd
from sklearn.metrics import confusion_matrix
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


# In[ ]:


login(token="YOUR_HF_TOKEN_HERE")


# ## 1. Data Load and Preprocessing

# In[2]:


dataset_path = "PATH_TO_YOUR_DATA"

with open(dataset_path, "r", encoding="utf-8") as f:
    ner_dataset = json.load(f)

print(f"Loaded {len(ner_dataset)} clinical notes")


# In[3]:


ner_dataset[0]


# In[5]:


TOKEN_RE = re.compile(r'\S+')

def tokenize_with_offsets(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

def process_item(item):
    text = item["data"]["comment"]
    tokens_data = tokenize_with_offsets(text)
    tokens = [t[0] for t in tokens_data]
    
    spans = []
    results = item["annotations"][0]["result"]
    for r in results:
        if r["type"] == "labels":
            v = r["value"]
            spans.append({
                "start": v["start"],
                "end": v["end"],
                "label": v["labels"][0]
            })
    
    labels = ["O"] * len(tokens)
    for i, (word, t_start, t_end) in enumerate(tokens_data):
        for span in spans:
            # Lógica de solapamiento (Overlap):
            # El token se etiqueta si hay una intersección entre sus caracteres y los de la etiqueta
            if max(t_start, span["start"]) < min(t_end, span["end"]):
                labels[i] = span["label"]
                break 
                
    return {"tokens": tokens, "labels": labels}

result = process_item(ner_dataset[0])
print(result)


# In[6]:


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
        
        # Extraer anotaciones de tipo 'labels'
        spans = []
        # Verificamos que existan anotaciones para evitar errores
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
        
        # Asignación con lógica de solapamiento
        labels = ["O"] * len(tokens)
        for i, (word, t_start, t_end) in enumerate(tokens_data):
            for span in spans:
                # Si el token y el span se tocan en al menos 1 carácter
                if max(t_start, span["start"]) < min(t_end, span["end"]):
                    labels[i] = span["label"]
                    break
        
        processed_data.append({
            "tokens": tokens,
            "labels": labels
        })
        
    return processed_data, sorted(list(unique_labels))

# 2. Ejecutamos el bucle
final_samples, label_list = process_ner_dataset(ner_dataset)

# 3. Creamos el mapeo de IDs (Indispensable para el modelo)
# 'O' siempre debe ser el ID 0
if "O" in label_list: label_list.remove("O")
label_list = ["O"] + label_list
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Procesados {len(final_samples)} ejemplos.")
print(f"Etiquetas encontradas: {label_list}")


# In[15]:


final_samples[700]


# ## 2. Train-test Split

# In[16]:


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


# ## 3. Fine Tune Llama3

# In[17]:


# fine tune with qlora llama3 8B
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

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

def format_example(note):
    tokens = note['tokens']
    labels = note['labels']

    entities = []
    current_entity = None

    for token, label in zip(tokens, labels):
        if label != "O":
            # Si ya hay una entidad en curso y es de la misma etiqueta, concatenamos
            if current_entity and current_entity["label"] == label:
               current_entity["text"] += f" {token}"
            else:
               # Si es una etiqueta nueva, creamos entidad nueva
               current_entity = {"text": token, "label": label}
               entities.append(current_entity)
        else:
            # Si el token es "O", cerramos la entidad actual
            current_entity = None

    instruction = "Identifica y extrae las entidades clínicas (Diagnosis, Treatment, Response, Date, Blasts, GeneMutation, VAF, ProteinChange, cDNAChange, Origin, Karyotype, Risk, FamilyHistory, Smoker, Exitus) del texto médico y devuélvelas en formato JSON."    
    input_text = " ".join(note['tokens'])
    response_text = json.dumps(entities, ensure_ascii=False)

    # IMPORTANTE: Añadir el token de finalización del tokenizer
    eos_token = tokenizer.eos_token

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{response_text}{eos_token}"
    )

    return {"text": full_prompt}

# convert to dataset from the lists
ds_train = Dataset.from_list(train_set).map(format_example)
ds_val = Dataset.from_list(val_set).map(format_example)
ds_test = Dataset.from_list(test_set).map(format_example)


# In[ ]:


# remove solution from train and val
ds_train = ds_train.remove_columns(['tokens', 'labels'])
ds_val = ds_val.remove_columns(['tokens', 'labels'])


# In[ ]:


sft_config = SFTConfig(
    output_dir="./llama3-v2-lora-ner",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5, #estaba a 2e-4
    save_strategy="epoch",
    logging_steps=10,
    eval_strategy="epoch",
    fp16=False,
    bf16=True, # Mejor para A40
    gradient_checkpointing=True,
    dataset_text_field="text", 
        packing=False,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds_train, 
    eval_dataset=ds_val,
    args=sft_config,
)

trainer.train()


# In[ ]:


# save adapter
OUT_DIR_LLM = "cr-multia/medical_ner/FINETUNNING/RESULTS_V2_confmatrix"
out = trainer.save_model(OUT_DIR_LLM)
out
print("Adapter would be saved to:", OUT_DIR_LLM)


# ## 4. Get Results

# In[ ]:

def normalize_text_spacing(text):
    """
    Inserta espacios alrededor de símbolos especiales para que la 
    tokenización sea consistente (ej: '42,95%' -> '42 , 95 %')
    """
    if not text: return ""
    # 1. Normalizar espacios (quitar dobles espacios, tabs, etc)
    text = " ".join(text.split())
    # 2. Poner espacios alrededor de cualquier símbolo no alfanumérico
    # Esto asegura que "c.524" se convierta en "c . 524"
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    # 3. Limpiar espacios dobles creados por el paso anterior
    return " ".join(text.split()).lower()

def generate_ner_report(model, tokenizer, test_set, save_path="predicciones_tfm.json"):
    all_true_labels = []
    all_pred_labels = []
    results = [] 
    
    model.eval()
    print("Iniciando inferencia en el set de test...")
    
    for example in tqdm(test_set):
        tokens = example['tokens']
        true_labels = example['labels']
        input_text = " ".join(tokens)

        # prompt definition
        instruction = "Identifica y extrae las entidades clínicas (Diagnosis, Treatment, Response, Date, Blasts, GeneMutation, VAF, ProteinChange, cDNAChange, Origin, Karyotype, Risk, FamilyHistory, Smoker, Exitus) del texto médico y devuélvelas en formato JSON."
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
        
        # generation of the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1, 
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id 
            )
        
        generated_response = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # mapping to our labels
        pred_labels = ["O"] * len(tokens)
        try:
            start_idx = generated_response.find("[")
            end_idx = generated_response.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                entities = json.loads(generated_response[start_idx:end_idx])
                
                for ent in entities:
                    # Normalizamos la predicción: "c.524G" -> "c . 524g"
                    ent_text_norm = normalize_text_spacing(ent['text'])
                    ent_label = ent['label']
                    
                    if not ent_text_norm: continue
                    
                    found = False
                    # Buscamos en los tokens originales
                    for i in range(len(tokens)):
                        for l in range(1, 35): # Ventana para cariotipos
                            if i + l > len(tokens): break
                            
                            # Normalizamos también la ventana de tokens originales
                            window_text_norm = normalize_text_spacing(" ".join(tokens[i:i+l]))
                            
                            if ent_text_norm == window_text_norm:
                                for j in range(l):
                                    pred_labels[i + j] = ent_label
                                found = True
                                break
                        if found: break 
        except Exception as e:
            pass

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        
        # save results
        results.append({
            "text": input_text,
            "true": true_labels,
            "pred": pred_labels,
            "raw_json": generated_response
        })

    report = classification_report(all_true_labels, all_pred_labels, digits=3)
    return report, results


# In[ ]:


report, results = generate_ner_report(model, tokenizer, test_set)
print(report)


# In[ ]:


# save results
with open("cr-multia/medical_ner/FINETUNNING/RESULTS_V2_confmatrix/llama3_results_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Results saved to llama3_results_v2.json")


# save report
with open("cr-multia/medical_ner/FINETUNNING/RESULTS_V2_confmatrix/medical_ner_report_LLAMA3_FINETUNNED_v2.txt", "w") as f:
    f.write("Medical NER Classification Report\n")
    f.write(report)

print("medical_ner_report_LLAMA3_FINETUNNED_v2.txt")

# Gather all true and predicted labels from results
all_true = [label for r in results for label in r["true"]]
all_pred = [label for r in results for label in r["pred"]]

# Build confusion matrix
cm = confusion_matrix(all_true, all_pred, labels=label_list)
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)

# Save to CSV
cm_path = "cr-multia/medical_ner/FINETUNNING/RESULTS_V2_confmatrix/confusion_matrix_llama3_v2.csv"
cm_df.to_csv(cm_path)
print(f"Confusion matrix saved to {cm_path}")
