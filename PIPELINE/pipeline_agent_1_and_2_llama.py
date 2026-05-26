# TEST PIPELINE 1: FROM AGENT 1 TO AGENT 2

# dependencies
import sys
import torch
import re
import json
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# path to agents
PATH_AGENT_1 = "/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/FINETUNNING/RESULTS_BERT_V1/RESULTS_BERT_V1/checkpoint-1290"
PATH_AGENT_2_BASE  = "meta-llama/Meta-Llama-3-8B-Instruct"
PATH_AGENT_2_LORA  = "/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/FINETUNNING/RESULTS_LLAMA"

# LABEL DEFINITIONS
ALLOWED_LABELS = ["Diagnosis", "Smoker", "GeneMutation", "Treatment", "Exitus", "FamilyHistory"]

VALID_ATTRIBUTES_MAP = {
    "Diagnosis":     ["Confirmed", "Suspicion", "Discarded", "Progression", "Control"],
    "Smoker":        ["Yes", "Previous", "No"],
    "Treatment":     ["Yes", "No"],
    "GeneMutation":  ["Yes", "No"],
    "Exitus":        ["Yes", "No"],
    "FamilyHistory": ["Yes", "No"]
}

OPTS_MAP = {
    "Diagnosis":     "[Confirmed, Control, Progression, Suspicion, Discarded]",
    "Smoker":        "[Yes, Previous, No]",
    "GeneMutation":  "[Yes, No]",
    "Treatment":     "[Yes, No]",
    "Exitus":        "[Yes, No]",
    "FamilyHistory": "[Yes, No]"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Charging models...")

# Agent 1: NER (Token Classification)
tokenizer_ner = AutoTokenizer.from_pretrained(PATH_AGENT_1, add_prefix_space=True)
model_ner = AutoModelForTokenClassification.from_pretrained(PATH_AGENT_1).to(device)
id2label_ner  = model_ner.config.id2label

# Agent 2: Attributes (Sequence Classification)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer_llm = AutoTokenizer.from_pretrained(PATH_AGENT_2_BASE)
tokenizer_llm.pad_token = tokenizer_llm.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    PATH_AGENT_2_BASE,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Lora weights on LLM
model_llm = PeftModel.from_pretrained(base_model, PATH_AGENT_2_LORA)
model_llm.eval()

print("Models loaded!")

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

def build_prompt(entity_info, full_context):
    options = OPTS_MAP.get(entity_info['label'], "[Yes, No]")
    
    system_prompt = f"""Eres un experto en codificacion medica. Tu objetivo es clasificar el estado de la entidad entre corchetes [...] dentro del contexto clinico.

REGLAS ESTRICTAS:
1. Responde UNICAMENTE con un objeto JSON.
2. La clave debe ser "attribute" y el valor una de estas opciones: [{options}].
3. No des explicaciones ni texto adicional.
"""
    
    user_prompt = f"""Analiza el atributo de la siguiente nota clinica segun el contexto:

Contexto:
{full_context}

Entidad: "{entity_info['word']}"

Label de la entidad: {entity_info['label']}

Atributo:"""

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def parse_llm_response(generated_text, entity_label):
    valid_options = VALID_ATTRIBUTES_MAP.get(entity_label, [])
    
    try:
        match = re.search(r'\"attribute\":\s*\"(\w+)\"', generated_text)
        if match:
            prediction = match.group(1)
            if prediction in valid_options:
                return prediction
    except:
        pass
    
    for opt in valid_options:
        if opt.lower() in generated_text.lower():
            return opt
    
    return "Parsing_Error"


def agent_2(text, entity_info):
    full_context = text[:entity_info['start']] + f"[{entity_info['word']}]" + text[entity_info['end']:]
    prompt       = build_prompt(entity_info, full_context)
    
    inputs = tokenizer_llm(prompt, return_tensors="pt").to(model_llm.device)
    
    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,          # greedy — igual que en evaluación
            temperature=1.0,          # ignorado con do_sample=False
            eos_token_id=tokenizer_llm.eos_token_id
        )
    
    generated_text = tokenizer_llm.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    prediction = parse_llm_response(generated_text, entity_info['label'])
    return prediction, generated_text  # devuelve también el raw para debug

# FINAL PIPELINE
def final_pipeline(text, debug=False):
    entities = agent_1(text)
    results  = []
    
    for ent in entities:
        attribute  = "N/A"
        raw_output = None
        
        if ent['label'] in ALLOWED_LABELS:
            attribute, raw_output = agent_2(text, ent)
        
        result = {
            "text":      ent["word"],
            "label":     ent["label"],
            "attribute": attribute,
            "offset":    (ent["start"], ent["end"])
        }
        
        if debug:
            result["llm_raw"] = raw_output
        
        results.append(result)
    
    return results


# TEST

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Provide a clinical note as argument.")
        sys.exit(1)
    
    input_text = sys.argv[1]
    debug_mode = "--debug" in sys.argv  # activa con: python script.py "texto" --debug
    
    final_output = final_pipeline(input_text, debug=debug_mode)
    print("Result!")
    print(json.dumps(final_output, indent=4, ensure_ascii=False))
