# TEST FULL PIPELINE ON TEST SET: FROM AGENT 1 TO AGENT 3

# dependencies
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from seqeval.scheme import IOB2
import sys
import torch
import re
import random
import json
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from pipeline_agent_1_to_3_llama import final_pipeline

DATA_PATH = "PATH_TO_YOUR_DATA"

OUTPUT_FILE = "test_results_full_pipeline.json"

def create_gold_standard(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # --- CRITICAL: Replicate the split from training ---
    random.seed(43)
    random.shuffle(raw_data)
    n_total = len(raw_data)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    test_data = raw_data[n_train + n_val:] # Only the 10% test set
    
    master_gold = []
    for item in test_data:
        text = item["data"]["comment"]
        annotations = item.get("annotations", [{}])[0].get("result", [])
        
        entities = {}
        relations = []
        choices = {}
        
        for r in annotations:
            if r["type"] == "labels":
                entities[r["id"]] = {
                    "id": r["id"],
                    "word": r["value"]["text"],
                    "label": r["value"]["labels"][0],
                    "start": r["value"]["start"],
                    "end": r["value"]["end"]
                }
            elif r["type"] == "choices":
                choices[r["id"]] = r["value"]["choices"][0]
            elif r["type"] == "relation":
                relations.append({
                    "from_id": r["from_id"], "to_id": r["to_id"], "type": r["labels"][0]
                })

        master_gold.append({
            "text": text,
            "gold_entities_raw": [r for r in annotations if r["type"] == "labels"], # For NER metric
            "gold_entities_processed": entities, # For ID mapping
            "gold_attributes": choices,
            "gold_relations": relations
        })
    return master_gold

def run_evaluation():
    test_samples = create_gold_standard(DATA_PATH)
    results = []

    print(f"Starting Integrated Pipeline Test on {len(test_samples)} UNSEEN samples...")

    for item in tqdm(test_samples):
        try:
            # The pipeline handles its own internal agent logic
            prediction = final_pipeline(item["text"])
        except Exception as e:
            print(f"Error: {e}")
            prediction = {"entities": [], "relations": []}

        results.append({
            "text": item["text"],
            "gold": {
                "entities": item["gold_entities_raw"],
                "attributes": item["gold_attributes"],
                "relations": item["gold_relations"]
            },
            "prediction": prediction
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def calculate_metrics(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Agent 1: NER Metrics (Span-based overlap)
    ner_true, ner_pred = [], []
    
    # Agent 2: Attribute Metrics (Only for correctly identified entities)
    attr_true, attr_pred = [], []

    # Agent 3: Relation Metrics (Based on entity types and relation label)
    rel_true, rel_pred = [], []

    for item in data:
        gold_ents = item["gold"]["entities"]
        pred_ents = item["prediction"]["entities"]
        
        # 1. NER
        gold_spans = {e["value"]["start"]: e["value"]["labels"][0] for e in gold_ents}
        pred_spans = {e["start"]: e["label"] for e in pred_ents}
        
        all_offsets = set(gold_spans.keys()) | set(pred_spans.keys())
        for offset in all_offsets:
            ner_true.append(gold_spans.get(offset, "O"))
            ner_pred.append(pred_spans.get(offset, "O"))

        # 2. Attribute Evaluation
        # We only evaluate attributes for entities where the NER was correct
        gold_attr_map = item["gold"]["attributes"] # Key: Label Studio ID
        
        # Map LS IDs to offsets for matching
        offset_to_gold_attr = {e["value"]["start"]: gold_attr_map.get(e["id"], "N/A") for e in gold_ents}
        
        for p_ent in pred_ents:
            if p_ent["start"] in offset_to_gold_attr:
                gold_attr = offset_to_gold_attr[p_ent["start"]]
                if gold_attr == "N/A":          # ← skip entities with no attribute annotation
                    continue
                attr_true.append(gold_attr)
                attr_pred.append(p_ent["attribute"])

        # 3. Relation Evaluation
        # Since IDs change, we match relations by: (From_Label, To_Label, Relation_Type)
        gold_id_to_label = {e["id"]: e["value"]["labels"][0] for e in gold_ents}
        g_rels = set([(gold_id_to_label.get(r["from_id"], "UNK"), 
               gold_id_to_label.get(r["to_id"], "UNK"), 
               r["type"]) for r in item["gold"]["relations"]])
        
        pred_id_to_label = {e["id"]: e["label"] for e in pred_ents}
        p_rels = set([(pred_id_to_label.get(r["from_id"]), pred_id_to_label.get(r["to_id"]), r["type"]) for r in item["prediction"]["relations"]])
        
        # For simplicity in this report, we treat every unique (Type1, Type2, Rel) as a sample
        # A more robust way is counting TP, FP, FN directly
        all_rel_combos = g_rels | p_rels
        for rel in all_rel_combos:
            rel_true.append(1 if rel in g_rels else 0)
            rel_pred.append(1 if rel in p_rels else 0)

    # Report
    print("\n" + "="*30)
    print("AGENT 1: NER CLASSIFICATION")
    print(classification_report(ner_true, ner_pred))

    print("\n" + "="*30)
    print("AGENT 2: ATTRIBUTE CLASSIFICATION")
    print(classification_report(attr_true, attr_pred))

    print("\n" + "="*30)
    print("AGENT 3: RELATION EXTRACTION")
    p, r, f, _ = precision_recall_fscore_support(rel_true, rel_pred, average='binary')
    print(f"Precision: {p:.2%}\nRecall: {r:.2%}\nF1-Score: {f:.2%}")

if __name__ == "__main__":
    # 1. Generate Predictions
    run_evaluation()
    
    # 2. Calculate and Print Metrics
    calculate_metrics(OUTPUT_FILE)

