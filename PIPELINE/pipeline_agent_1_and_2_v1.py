# TEST PIPELINE 1: FROM AGENT 1 TO AGENT 2

# dependencies
import sys
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

# path to agents
PATH_AGENT_1 = "/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/FINETUNNING/RESULTS_BERT_V1/RESULTS_BERT_V1/checkpoint-1290"
PATH_AGENT_2 = "/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/FINETUNNING/RESULTS_BERT"

# list definitions
ALLOWED_LABELS = ["Diagnosis", "Smoker", "GeneMutation", "Treatment", "Exitus", "FamilyHistory"]

VALID_ATTRIBUTES_MAP = {
    "Diagnosis": ["Confirmed", "Suspicion", "Discarded", "Progression", "Control"],
    "Smoker": ["Yes", "No", "Previous"],
    "Treatment": ["Yes", "No"],
    "GeneMutation": ["Yes", "No"],
    "Exitus": ["Yes", "No"],
    "FamilyHistory": ["Yes", "No"]
}

# use GPU for faster response
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Charging models...")

# Agent 1: NER (Token Classification)
tokenizer_ner = AutoTokenizer.from_pretrained(PATH_AGENT_1, add_prefix_space=False)
model_ner = AutoModelForTokenClassification.from_pretrained(PATH_AGENT_1).to(device)

# Agent 2: Attributes (Sequence Classification)
tokenizer_attr = AutoTokenizer.from_pretrained(PATH_AGENT_2, add_prefix_space=True)
model_attr = AutoModelForSequenceClassification.from_pretrained(PATH_AGENT_2).to(device)

# label mapping
id2label_ner = model_ner.config.id2label
id2label_attr = model_attr.config.id2label
print(id2label_ner)
print(id2label_attr)

# Get entities using Agent 1
def agent_1(text):

    print(f"\n[DEBUG NER] Input text received: '{text}'")
    inputs = tokenizer_ner(
        text, 
        return_tensors="pt", 
        truncation=True, 
        return_offsets_mapping=True
    ).to(device)
    
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()    
    
    with torch.no_grad():
        outputs = model_ner(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

    all_labels_pred = [id2label_ner[p] for p in predictions]
    non_o_preds = [l for l in all_labels_pred if l != 'O']
    print(f"[DEBUG NER] Total tokens: {len(all_labels_pred)} | Non-'O' predictions: {len(non_o_preds)}")
    if len(non_o_preds) > 0:
        print(f"[DEBUG NER] Labels found: {set(non_o_preds)}")
    
    raw_entities = []
    for i, pred_id in enumerate(predictions):
        label_name = id2label_ner[pred_id]
        start, end = offset_mapping[i]
        if label_name != "O" and (start != 0 or end != 0):
            raw_entities.append({
                "word": text[start:end],
                "label": label_name,
                "start": int(start),
                "end": int(end)
            })

    if not raw_entities:
        return []

    merged_entities = []
    if raw_entities:
        current_ent = raw_entities[0]

        for next_ent in raw_entities[1:]:
            if next_ent["label"] == current_ent["label"] and (next_ent["start"] <= current_ent["end"] + 1):
                current_ent["end"] = next_ent["end"]
                current_ent["word"] = text[current_ent["start"]:current_ent["end"]]
            else:
                merged_entities.append(current_ent)
                current_ent = next_ent
        
        merged_entities.append(current_ent)

    for ent in merged_entities:
        ent["word"] = ent["word"].strip()

    return merged_entities

# Classify attributes with Agent 2
def agent_2(text, entity_info):

    full_context = text[:entity_info['start']] + f"[{entity_info['word']}]" + text[entity_info['end']:]
    
    inputs = tokenizer_attr(
        full_context,           # examples["context"]
        entity_info['word'],    # examples["entity_text"]
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        logits = model_attr(**inputs).logits
    
    probs = torch.softmax(logits, dim=1)[0]
    
    allowed_attrs = VALID_ATTRIBUTES_MAP.get(entity_info['label'], [])
    
    best_attr = "Not applicable"
    max_p = -1.0
    
    if allowed_attrs:
        for idx, attr_name in id2label_attr.items():
            if attr_name in allowed_attrs:
                p_actual = probs[idx].item()
                if p_actual > max_p:
                    max_p = p_actual
                    best_attr = attr_name
    else:
        best_idx = torch.argmax(probs).item()
        best_attr = id2label_attr[best_idx]
        max_p = probs[best_idx].item()

    return best_attr, max_p

def pipeline(text):
    
    entities = agent_1(text)
    
    results = []
    
    for ent in entities:
        attribute = "N/A"
        score = None
        
        if ent['label'] in ALLOWED_LABELS:
            attribute, score = agent_2(text, ent)
            score = round(score, 4)
            
        results.append({
            "text": ent["word"],
            "label": ent["label"],
            "attribute": attribute,
            "confidence_score": score,
            "offset": (ent["start"], ent["end"])
        })
            
    return results

# TEST

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Provide a clinical note using double commas.")
        sys.exit(1)
    
    input_text = sys.argv[1]
    
    final_output = pipeline(input_text)
    print("Result!")
    print(json.dumps(final_output, indent=4, ensure_ascii=False))
