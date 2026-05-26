# TEST PIPELINE 1: FROM AGENT 1 TO AGENT 2

# dependencies
import sys
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline

# path to agents
PATH_AGENT_1 = "/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/FINETUNNING/RESULTS_BERT_V1/RESULTS_BERT_V1/checkpoint-1290"
PATH_AGENT_2 = "/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/FINETUNNING/RESULTS_BERT_v3"

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

unique_attributes = ["Confirmed", "Control", "Progression", "Suspicion", "Discarded", "Yes", "Previous", "No"]
manual_id2label = {i: label for i, label in enumerate(unique_attributes)}

# use GPU for faster response
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Charging models...")

# Agent 1: NER (Token Classification)
tokenizer_ner = AutoTokenizer.from_pretrained(PATH_AGENT_1, add_prefix_space=True)
model_ner = AutoModelForTokenClassification.from_pretrained(PATH_AGENT_1).to(device)

# Agent 2: Attributes (Sequence Classification)
tokenizer_attr = AutoTokenizer.from_pretrained(PATH_AGENT_2, add_prefix_space=True)
model_attr = AutoModelForSequenceClassification.from_pretrained(PATH_AGENT_2).to(device)

# label mapping
id2label_ner = model_ner.config.id2label
id2label_attr = model_attr.config.id2label
print(id2label_ner)
print(id2label_attr)

print("Model id2label:", model_attr.config.id2label)
print("Manual id2label:", manual_id2label)

# Get entities using Agent 1
TOKEN_RE = re.compile(r'\w+|[^\w\s]')

def tokenize_with_offsets(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

ner_pipeline_debug = pipeline(
    "token-classification",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="none"  # ← sin agregación
)

def debug_tokenization(text):
    raw_tokens = ner_pipeline_debug(text)
    
    print("=== Tokens raw del NER ===")
    for tok in raw_tokens:
        if tok['entity'] != 'O':  # solo entidades, ignora O
            print(f"  token: {repr(tok['word']):<20} label: {tok['entity']:<20} score: {tok['score']:.4f}")

def agent_1(text):
    # tokenize like training
    tokens_data = tokenize_with_offsets(text)
    tokens = [t[0] for t in tokens_data]
    
    if not tokens:
        return []
    
    encoding = tokenizer_ner(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # inference
    with torch.no_grad():
        logits = model_ner(**encoding).logits
    
    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    word_ids = encoding.word_ids(batch_index=0)
    
    # group entities
    word_labels = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_labels:  # solo el primer subword de cada palabra
            word_labels[word_idx] = id2label_ner[predictions[token_idx]]
    
    results = []
    current = None
    
    for word_idx, (word, w_start, w_end) in enumerate(tokens_data):
        label = word_labels.get(word_idx, "O")
        
        if label == "O":
            if current:
                results.append(current)
                current = None
            continue
        
        if current is None:
            current = {"word": word, "label": label, "start": w_start, "end": w_end}
        elif label == current["label"]:
            current["word"] = text[current["start"]:w_end]
            current["end"] = w_end
        else:
            results.append(current)
            current = {"word": word, "label": label, "start": w_start, "end": w_end}
    
    if current:
        results.append(current)
    
    return results

# Classify attributes with Agent 2
def agent_2(text, entity_info):

    full_context = text[:entity_info['start']] + f"[{entity_info['word']}]" + text[entity_info['end']:]
    
    inputs = tokenizer_attr(
        full_context,         
        entity_info['word'],  
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        logits = model_attr(**inputs).logits
    
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    model_id2label = model_attr.config.id2label
    
    allowed_attrs = VALID_ATTRIBUTES_MAP.get(entity_info['label'], [])
    
    allowed_probs = {}

    for i in range(len(probs)):
        attr_name = model_id2label.get(i) or model_id2label.get(str(i))
        if attr_name in allowed_attrs:
            allowed_probs[attr_name] = float(probs[i])
    
    if not allowed_probs:
        return "Not applicable", 0.0
    
    total = sum(allowed_probs.values())
    normalized = {attr: p / total for attr, p in allowed_probs.items()}
    
    best_attr = max(normalized, key=normalized.get)
    best_score = normalized[best_attr]
    
    return best_attr, best_score

def final_pipeline(text):
    
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
            "confidence_score": round(float(score), 4) if score is not None else None,
            "offset": (ent["start"], ent["end"])
        })
            
    return results

# TEST

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Provide a clinical note using double commas.")
        sys.exit(1)
    
    input_text = sys.argv[1]
    
    debug_tokenization(input_text)

    final_output = final_pipeline(input_text)
    print("Result!")
    print(json.dumps(final_output, indent=4, ensure_ascii=False))
