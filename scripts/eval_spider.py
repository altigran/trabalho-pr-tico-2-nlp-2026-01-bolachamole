"""Avalia um modelo no Spider DEV split com a métrica de Execution Accuracy (Fases 2 e 4).

MESMO procedimento para baseline (modelo base) e fine-tuned. O prompt few-shot é FIXO:
esquema do banco-alvo + pergunta + 3 exemplos (esquema + pergunta + SQL) do train split.

Geração DETERMINÍSTICA: greedy, temperature = 0.

Salva as predições (saída bruta + db_id + gold) para serem avaliadas pela métrica/pytest.

    python scripts/eval_spider.py --model <hf_id_ou_path> --split dev --out results/baseline/
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from peft import PeftModel
from preprocess_spider import serialize_schema


FEWSHOT_PROMPT_TEMPLATE = [{"role": "system", "content": "Dado um esquema e uma pergunta, escreva uma consulta SQL que a responde."}]
with open("data/processed/train_chat.jsonl", 'r') as arq:
  for i in range(3):
    linha = arq.readline()
    exemplo = json.loads(linha.strip())
    FEWSHOT_PROMPT_TEMPLATE.append({"role": "user", "content": exemplo["user"]})
    FEWSHOT_PROMPT_TEMPLATE.append({"role": "assistant", "content": exemplo["assistant"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="caminho do adapter LoRA (fine-tuned)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/baseline/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    #  1. set_seed; carregar modelo (+ adapter se fine-tuned) e tokenizer
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

    #  2. para cada exemplo do dev: montar prompt few-shot e gerar com greedy/temp=0
    #  3. salvar results/.../predictions.jsonl com {input, actual_output, expected_output, db_id}
    with open(f"data/spider/{args.split}.json", 'r') as arq:
      split = json.load(arq)

    with open(f"data/spider/tables.json", 'r') as arq:
      tables = json.load(arq)
    tabelas = {}
    for item in tables:
      tabelas[item["db_id"]] = item

    with open(f"{args.out}/predictions.jsonl", 'w') as arq:
      i = 1
      for item in split:
        print(f"Respondendo item {i} de {len(split)}")
        db_id = item["db_id"]
        expected_output = item["query"]
        input_q = item["question"]
        prompt = list(FEWSHOT_PROMPT_TEMPLATE)
        prompt.append({"role": "user", "content": f"Schema: {serialize_schema(tabelas[db_id])}\nQuestion: {input_q}"})

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
        generated_tokens = output[0][input_ids_tensor.shape[1]:]
        predicao = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predicao_sem_newlines = predicao.replace("\n", " ")
        arq.write(f"{json.dumps({'input': input_q, 'actual_output': predicao_sem_newlines, 'expected_output': expected_output, 'db_id': db_id})}\n")
        i+=1

    print(f"Predições salvas em {args.out}/predictions.jsonl")

    #  4. (opcional) computar Execution Accuracy aqui e salvar o resumo agregado
    '''
    metrica = ExecutionAccuracy(db_root="data/spider/database")
    num = 0
    sucessos = 0
    falhas = 0

    with open(f"{args.out}/predictions.jsonl", 'r') as arq:
      for linha in arq:
        num += 1
        item = json.loads(linha)
        teste = LLMTestCase(
          input=item["input"],
          actual_output=item["actual_output"],
          expected_output=item["expected_output"],
          additional_metadata=item["db_id"]
        )
        result = metrica.measure(teste)
        if (result == 1.0):
          sucessos += 1
        else:
          falhas += 1
    resumo = {
        "consultas totais:" num,
        "sucessos": sucessos,
        "falhas": falhas,
        "acurácia": sucessos/num
    }
    with open(f"{args.out}/resumo.json", 'w') as arq:
      json.dump(resumo, arq)
    '''

if __name__ == "__main__":
    main()
