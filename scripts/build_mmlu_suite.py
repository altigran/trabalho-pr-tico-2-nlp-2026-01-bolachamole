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
    # TODO: baixar MMLU do HF Hub, amostrar 50 por categoria (total 150) com a seed,
    # separar 5 exemplos de few-shot e salvar os arquivos.
    raise NotImplementedError


if __name__ == "__main__":
    main()
