"""Avalia um modelo na suíte MMLU de 150 questões em modo 5-shot (Fase 5).

Usa os MESMOS 5 exemplos de contexto (data/mmlu/fewshot_5.jsonl) para o modelo base e
para os fine-tuned. Geração determinística (greedy, temperature = 0).

Reporta acurácia AGREGADA e POR CATEGORIA (STEM, Humanidades, Ciências Sociais) e salva
em results/mmlu/ para o cálculo de variação percentual base × fine-tuned.

    python scripts/eval_mmlu.py --model <hf_id_ou_path> --out results/mmlu/
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="caminho do adapter LoRA (fine-tuned)")
    ap.add_argument("--suite", default="data/mmlu/suite_150.jsonl")
    ap.add_argument("--fewshot", default="data/mmlu/fewshot_5.jsonl")
    ap.add_argument("--out", default="results/mmlu/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    #  1. set_seed; carregar modelo (+ adapter) e tokenizer
    set_seed(args.seed)

    if (args.adapter):
      quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
      model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto", quantization_config=quantization_config)
      model = PeftModel.from_pretrained(model, args.adapter)
    else:
      model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if (tokenizer.pad_token == None):
      tokenizer.pad_token = tokenizer.eos_token

    #  2. para cada questão: montar prompt 5-shot (A/B/C/D) e prever a alternativa
    messages = [{"role": "system", "content": "Retorne a alternativa correta entre A, B, C e D."}]
    with open(args.fewshot, 'r') as arq:
      for linha in arq:
        exemplo = json.loads(linha)
        content = f'Question: {exemplo["question"]}\nChoices:\nA) {exemplo["choices"][0]}\nB) {exemplo["choices"][1]}\nC) {exemplo["choices"][2]}\nD) {exemplo["choices"][3]}'
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": exemplo["answer"]})

    acertou = 0
    categorias = {"STEM": 0, "Humanas": 0, "Ciências Sociais": 0}
    with open(args.suite, 'r') as arq:
      for linha in arq:
        prompt = messages
        questao = json.loads(linha)
        content = f'Question: {questao["question"]}\nChoices:\nA) {questao["choices"][0]}\nB) {questao["choices"][1]}\nC) {questao["choices"][2]}\nD) {questao["choices"][3]}'
        prompt.append({"role": "system", "content": "Retorne a alternativa correta entre A, B, C e D."})
        prompt.append({"role": "user", "content": content})

        input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        input_ids_tensor = input_ids["input_ids"]
        attention_mask = input_ids["attention_mask"]

        with torch.no_grad():
          output = model.generate(
              input_ids=input_ids_tensor,
              attention_mask=attention_mask,
              do_sample=False, # greedy
              # temperature=0.0, o do_sample já cuida
              max_new_tokens=512,
              pad_token_id=tokenizer.pad_token_id
          )
        generated_tokens = output[0, -1]

        predicao = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        resp = questao["answer"].strip().upper()
        if (predicao == resp):
            acertou += 1
            categorias[questao["subject"]] += 1

    #  3. acurácia por categoria e agregada -> results/mmlu/<model>.json
    acur_agregada = acertou/150
    acur_categoria = {}
    for cat, certas in categorias.items():
      acur_categoria[cat] = certas/50

    resultados = {
        "corretas agreagada": acertou,
        "acurácia agregada": acur_agregada,
        "corretas por categoria": categorias,
        "acurácia por categoria": acur_categoria
    }

    nome_arq = args.model.replace('/', '_').replace('.', '_')
    if (args.adapter):
      nome_arq += "_fine_tuned"
    with open(f"{args.out}/{nome_arq}.json", 'w') as arq:
      json.dump(resultados, arq)

if __name__ == "__main__":
    main()
