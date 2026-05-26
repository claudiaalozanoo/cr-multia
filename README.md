# cr-multia

## Project Overview
This repository contains the core methodology, architecture, and logic for my project on **A LLM Multi-Agent System for Extraction and Interpretation of Myelodysplastic Syndromes Clinical Records**. 

The main objective of this project is to design, implement, and evaluate a LLM multi-agent framework for the automated structuring of complex clinical records in Myeloid Neoplasms, comparing Generative vs. Discriminative approaches for each specific task.

> ⚠️ **Scope:** This repository focuses on the **methodological framework, agent definitions, prompts, and workflows**. In compliance with data privacy regulations (GDPR/HIPAA) and GitHub storage limits, this repository **does not** contain real patient data, fine-tuned model weights, or LoRA adapters.

---

## Methodology & Agent Architecture

The system splits complex clinical NLP tasks into three specialized roles. 

### Agent Roles:
*   **Named Entity Recognition:** Identifies clinical entities and classifies them into the defined taxonomy.
*   **Attribute Classification:** Examines the context of the detected entity and defines an attribute or status related to that specific entity.
*   **Relation Extraction:** Stablishes connections or relationships between entities and classifies them.

![System Workflow Diagram](general_pipeline.png)

---

## Repository Structure

```text
.
├── medical_ner/                 # Agent 1: Named Entity Recognition
│   ├── FEW_SHOT/                # Baseline Benchmark of Generative Models Using Few-Shot Prompting Technique
│   └── FINETUNNING/             # Fine-tuning the Best Performing Generative Model and the Discriminative one
├── attribute_association/       # Agent 2: Attribute Association
│   ├── FEW_SHOT/                # Baseline Benchmark of Generative Models Using Few-Shot Prompting Technique
│   └── FINETUNNING/             # Fine-tuning the Best Performing Generative Model and the Discriminative one
├── relation_extraction/         # Agent 3: Relation Extraction
│   ├── FEW_SHOT/                # Baseline Benchmark of Generative Models Using Few-Shot Prompting Technique
│   └── FINETUNNING/             # Fine-tuning the Best Performing Generative Model and the Discriminative one
├── data/                        # 2,566 free-text clinical notes (Dataset splits for few-shot testing)
└── PIPELINE/                    # End-to-end script to test the full multi-agent pipeline
```

## Installation & Setup

### Prerequisites
- Python 3.10+
