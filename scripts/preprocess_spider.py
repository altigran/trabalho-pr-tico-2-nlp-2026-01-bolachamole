"""Pré-processa o Spider (train split) para o formato de chat do framework de treino.

Cada exemplo de treino deve conter, no prompt: a representação do ESQUEMA do banco-alvo
(tabelas, colunas, chaves) + a pergunta em linguagem natural; e, como resposta, o SQL gold.

Saída sugerida: data/processed/train_chat.jsonl

Lembre de FIXAR a seed para qualquer amostragem/embaralhamento.
"""

import argparse


def serialize_schema(db_schema) -> str:
    """TODO: transformar o esquema do banco em texto para o prompt (schema linking)."""
    raise NotImplementedError


def build_chat_example(question: str, schema_text: str, gold_sql: str) -> dict:
    """TODO: montar o exemplo no formato de chat (system/user/assistant ou [INST]...)."""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_root", default="data/spider")
    ap.add_argument("--out", default="data/processed/train_chat.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # TODO: implementar o pré-processamento.
    raise NotImplementedError


if __name__ == "__main__":
    main()
