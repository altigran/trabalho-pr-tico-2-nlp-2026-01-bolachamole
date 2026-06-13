"""Pré-processa o Spider (train split) para o formato de chat do framework de treino.

Cada exemplo de treino deve conter, no prompt: a representação do ESQUEMA do banco-alvo
(tabelas, colunas, chaves) + a pergunta em linguagem natural; e, como resposta, o SQL gold.

Saída sugerida: data/processed/train_chat.jsonl

Lembre de FIXAR a seed para qualquer amostragem/embaralhamento.
"""

import argparse
import json
import random
from transformers import set_seed
from pathlib import Path


def serialize_schema(db_schema) -> str:
    """Transforma o esquema do banco em texto para o prompt (schema linking)."""

    texto = "Tables: "
    tables = db_schema["table_names_original"]
    columns = db_schema["column_names_original"]

    colunas = {i: [] for i in range(len(tables))}
    for col in columns:
      id = col[0]
      if (id != -1):
        colunas[id].append(col[1])

    tabelas = []
    for i, tabela in enumerate(tables):
      tabelas.append(f"{tabela}({', '.join(colunas[i])})")
    texto += '; '.join(tabelas)

    texto += "\nPrimary keys: "
    chaves = []
    for i in db_schema["primary_keys"]:
      tabela, col = columns[i]
      chaves.append(f"{tables[tabela]}.{col}")
    texto += ', '.join(chaves)

    texto += "\nForeign keys: "
    chaves = []
    for i, j in db_schema["foreign_keys"]:
      tabela, col = columns[i]
      tabela2, col2 = columns[j]
      chaves.append(f"{tables[tabela]}.{col} -> {tables[tabela2]}.{col2}")
    texto += ', '.join(chaves)

    return texto + '\n'


def build_chat_example(question: str, schema_text: str, gold_sql: str) -> dict:
    """Monta o exemplo no formato de chat (system/user/assistant ou [INST]...)."""
    exemplo = {}
    exemplo["user"] = f"Schema: {schema_text}\nQuestion: {question}"
    exemplo["assistant"] = gold_sql
    return exemplo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_root", default="data/spider")
    ap.add_argument("--out", default="data/processed/train_chat.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    with open(f"{args.spider_root}/tables.json", 'r') as arq:
      tables = json.load(arq)
    tabelas = {}
    for item in tables:
      tabelas[item["db_id"]] = item

    with open(f"{args.spider_root}/train_spider.json", 'r') as arq:
      split = json.load(arq)
    random.shuffle(split)

    output_file = Path(args.out)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w') as arq:
      for item in split:
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        schema_text = serialize_schema(tabelas[db_id])
        exemplo = build_chat_example(question, schema_text, gold_sql)
        arq.write(f"{json.dumps(exemplo)}\n")

    print("Concluído. Salvo em", args.out)


if __name__ == "__main__":
    main()
