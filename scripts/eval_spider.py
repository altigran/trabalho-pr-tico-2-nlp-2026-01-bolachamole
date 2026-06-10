"""Avalia um modelo no Spider DEV split com a métrica de Execution Accuracy (Fases 2 e 4).

MESMO procedimento para baseline (modelo base) e fine-tuned. O prompt few-shot é FIXO:
esquema do banco-alvo + pergunta + 3 exemplos (esquema + pergunta + SQL) do train split.

Geração DETERMINÍSTICA: greedy, temperature = 0.

Salva as predições (saída bruta + db_id + gold) para serem avaliadas pela métrica/pytest.

    python scripts/eval_spider.py --model <hf_id_ou_path> --split dev --out results/baseline/
"""

import argparse


FEWSHOT_PROMPT_TEMPLATE = """\
TODO: definir o template FIXO de prompt few-shot, contendo:
  - o esquema do banco-alvo;
  - 3 exemplos (esquema + pergunta + SQL) extraídos do train split;
  - a pergunta atual.
Este template deve ser idêntico em todas as avaliações.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="caminho do adapter LoRA (fine-tuned)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/baseline/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # TODO:
    #  1. set_seed; carregar modelo (+ adapter se fine-tuned) e tokenizer
    #  2. para cada exemplo do dev: montar prompt few-shot e gerar com greedy/temp=0
    #  3. salvar results/.../predictions.jsonl com {input, actual_output, expected_output, db_id}
    #  4. (opcional) computar Execution Accuracy aqui e salvar o resumo agregado
    raise NotImplementedError


if __name__ == "__main__":
    main()
