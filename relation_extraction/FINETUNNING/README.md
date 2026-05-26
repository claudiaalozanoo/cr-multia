v1 -> normal inicial
v2 -> ajuste de parametros de lora 
v3 -> quitando el has_change_in (r y epochs)
v4 -> ajuste de mas parametros del finetunning (lora_alpha y learning rate)
v5 -> train test split debug added from V2
v6 -> derived from v5 , adaptation of parameters and tokenization errors (Claude): added collator so the model learns only the responses (since the prompt is too long)
1🔴 Crítico Usar offsets reales del JSON en format_text_with_ids
2🔴 Crítico Unificar pipeline de test con format_re_prompt
3🔴 Crítico Añadir tokenizer.padding_side = "left"
4🟡 Medio Añadir packing=False en SFTConfig
5🟡 Medio Verificar template de Qwen con turno assistant
8🟡 Medio Cambiar a lora_alpha=128 (= 2×r)
