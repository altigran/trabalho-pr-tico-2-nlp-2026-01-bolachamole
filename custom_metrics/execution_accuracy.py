"""Métrica customizada de Execution Accuracy para a tarefa Text-to-SQL (Spider).

Esta é a ÚNICA ferramenta de avaliação da tarefa-alvo e deve ser aplicada de forma
idêntica no baseline (Fase 2) e nos modelos fine-tuned (Fase 4).

VOCÊ deve implementar a lógica abaixo. A assinatura e o contrato estão fixados pelo
enunciado (Fase 1) — não altere a interface pública (`measure`, retorno float 1.0/0.0).
"""

import re
import sqlite3

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def extract_sql(raw_output: str) -> str:
    """Extrai a consulta SQL da saída BRUTA do modelo.

    Deve ser robusta a:
      - blocos markdown (```sql ... ``` ou ``` ... ```);
      - texto explicativo antes/depois da consulta;
      - prefixos como "SQL:", "Resposta:", etc.

    TODO: implementar. Retornar a string SQL limpa, pronta para execução.
    """
    raise NotImplementedError


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
        # TODO: implementar conforme o contrato acima.
        # Dica: o db_id do exemplo costuma vir em test_case.additional_metadata.
        # Lembre de fechar a conexão e de tratar exceções de sintaxe/execução.
        raise NotImplementedError

    def is_successful(self) -> bool:
        return getattr(self, "score", 0.0) >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"
