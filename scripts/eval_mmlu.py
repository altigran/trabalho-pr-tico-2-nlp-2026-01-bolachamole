"""Avalia um modelo na suíte MMLU de 150 questões em modo 5-shot (Fase 5).

Usa os MESMOS 5 exemplos de contexto (data/mmlu/fewshot_5.jsonl) para o modelo base e
para os fine-tuned. Geração determinística (greedy, temperature = 0).

Reporta acurácia AGREGADA e POR CATEGORIA (STEM, Humanidades, Ciências Sociais) e salva
em results/mmlu/ para o cálculo de variação percentual base × fine-tuned.

    python scripts/eval_mmlu.py --model <hf_id_ou_path> --out results/mmlu/
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="caminho do adapter LoRA (fine-tuned)")
    ap.add_argument("--suite", default="data/mmlu/suite_150.jsonl")
    ap.add_argument("--fewshot", default="data/mmlu/fewshot_5.jsonl")
    ap.add_argument("--out", default="results/mmlu/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # TODO:
    #  1. set_seed; carregar modelo (+ adapter) e tokenizer
    #  2. para cada questão: montar prompt 5-shot (A/B/C/D) e prever a alternativa
    #  3. acurácia por categoria e agregada -> results/mmlu/<model>.json
    raise NotImplementedError


if __name__ == "__main__":
    main()
