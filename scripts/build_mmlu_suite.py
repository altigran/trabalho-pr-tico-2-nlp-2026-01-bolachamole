"""Monta a suíte de avaliação MMLU: EXATAMENTE 150 questões, divididas igualmente em
3 categorias (50 cada):

    - STEM           (ex.: subcategoria computer_science)
    - Humanidades    (ex.: subcategoria philosophy)
    - Ciências Sociais (ex.: subcategoria economics)

Também seleciona os 5 exemplos de contexto (5-shot) que serão usados de forma IDÊNTICA
para o modelo base e para os fine-tuned.

FIXE a seed para que a amostragem das 150 questões e dos 5-shot seja reprodutível.
Saída sugerida: data/mmlu/suite_150.jsonl  e  data/mmlu/fewshot_5.jsonl
"""

import argparse
import json
from datasets import load_dataset, concatenate_datasets
from transformers import set_seed
from pathlib import Path

# Defina aqui as subcategorias escolhidas (uma por área).
SUBJECTS = {
    "STEM": "college_computer_science",
    "Humanities": "philosophy",
    "SocialSciences": "high_school_macroeconomics",
}
N_PER_CATEGORY = 50
N_FEWSHOT = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_suite", default="data/mmlu/suite_150.jsonl")
    ap.add_argument("--out_fewshot", default="data/mmlu/fewshot_5.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # baixa MMLU do HF Hub, amostrar 50 por categoria (total 150) com a seed,
    set_seed(args.seed)

    stem = load_dataset("cais/mmlu", SUBJECTS["STEM"], split="test")
    stem_50 = stem.shuffle(seed=args.seed).take(N_PER_CATEGORY)
    human = load_dataset("cais/mmlu", SUBJECTS["Humanities"], split="test")
    human_50 = human.shuffle(seed=args.seed).take(N_PER_CATEGORY)
    socsci = load_dataset("cais/mmlu", SUBJECTS["SocialSciences"], split="test")
    socsci_50 = socsci.shuffle(seed=args.seed).take(N_PER_CATEGORY)

    output_file = Path(args.out_suite)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out_suite, 'w') as arq:
      for dicionario in stem_50:
        dicionario["subject"] = "STEM"
        arq.write(f"{json.dumps(dicionario)}\n")
      for dicionario in human_50:
        dicionario["subject"] = "Humanas"
        arq.write(f"{json.dumps(dicionario)}\n")
      for dicionario in socsci_50:
        dicionario["subject"] = "Ciências Sociais"
        arq.write(f"{json.dumps(dicionario)}\n")

    # separa 5 exemplos de few-shot e salvar os arquivos.
    output_file = Path(args.out_fewshot)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out_fewshot, 'w') as arq:
      todos = concatenate_datasets([stem, human, socsci]).shuffle(seed=args.seed).take(N_FEWSHOT)
      for dicionario in todos:
        arq.write(f"{json.dumps(dicionario)}\n")

    print(f"Dataset MMLU salvo!")

if __name__ == "__main__":
    main()
