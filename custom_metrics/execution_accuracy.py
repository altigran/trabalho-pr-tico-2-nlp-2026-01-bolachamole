"""Métrica customizada de Execution Accuracy para a tarefa Text-to-SQL (Spider).

Esta é a ÚNICA ferramenta de avaliação da tarefa-alvo e deve ser aplicada de forma
idêntica no baseline (Fase 2) e nos modelos fine-tuned (Fase 4).
"""

import re
import sqlite3
import sqlparse

from collections import Counter
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def extract_sql(raw_output: str) -> str:
    """Extrai a consulta SQL da saída BRUTA do modelo.

    Deve ser robusta a:
      - blocos markdown (```sql ... ``` ou ``` ... ```);
      - texto explicativo antes/depois da consulta;
      - prefixos como "SQL:", "Resposta:", etc.

    Retorna a string SQL limpa, pronta para execução.
    """
    regex = r"```(?:sql)?\s*(.*?)\s*```"
    output = re.search(regex, raw_output, re.DOTALL | re.IGNORECASE)
    if (output):
        return output.group(1).strip()
    regex = r"SQL:\s*(.*?)"
    output = re.search(regex, raw_output, re.DOTALL | re.IGNORECASE)
    if (output):
        return output.group(1).strip()

    return raw_output.strip()


class ExecutionAccuracy(BaseMetric):
    """Compara o resultado da execução do SQL gerado com o do SQL de referência.

    Contrato (enunciado, Fase 1):
      a. extração robusta do SQL a partir de `test_case.actual_output`;
      b. conexão com o SQLite da base de teste do Spider correspondente ao exemplo;
      c. executar o SQL gerado em transação segura (try/except para erro de sintaxe);
      d. executar o SQL de referência (`test_case.expected_output`);
      e. comparar os conjuntos de resultados de forma INSENSÍVEL à ordem das linhas,
         EXCETO quando a consulta contiver ORDER BY (aí a ordem deve ser respeitada).
         Referência de semântica: avaliador oficial do Spider (test-suite evaluation);
      f. retornar 1.0 (resultados idênticos) ou 0.0 (falha/erro/divergência).
    """

    def __init__(self, db_root: str, threshold: float = 1.0):
        # db_root: diretório com os SQLite do Spider (ex.: data/spider/database/<db_id>/<db_id>.sqlite)
        self.db_root = db_root
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        db_id = test_case.additional_metadata
        try:
          with sqlite3.connect(self.db_root + f"/{db_id}/{db_id}.sqlite") as con:
            cur = con.cursor()

            # transforma em tuplas do sqlparse para normalizar antes de executar
            resp_sql = sqlparse.parse(extract_sql(test_case.actual_output))
            gaba_sql = sqlparse.parse(test_case.expected_output)
            resposta = sqlparse.format(resp_sql, keyword_case='lower', strip_comments=True)
            gabarito = sqlparse.format(gaba_sql, keyword_case='lower', strip_comments=True)

            cur.execute(resposta)
            res1 = cur.fetchall()
            cur.execute(gabarito)
            res2 = cur.fetchall()

            if ("order by" not in gabarito):
              if (Counter(res1) != Counter(res2)):
                return 0.0
            elif ("order by" in gabarito) and (res1 != res2):
              return 0.0

            cur.close()
            return 1.0
        except Exception as erro:
          print(erro)
        return 0.0

    def is_successful(self) -> bool:
        return getattr(self, "score", 0.0) >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"
