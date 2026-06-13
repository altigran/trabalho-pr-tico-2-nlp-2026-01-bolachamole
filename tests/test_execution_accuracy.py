"""Avaliação automatizada da métrica de Execution Accuracy via pytest (Fase 4).

Carrega as predições salvas (baseline ou fine-tuned), monta os LLMTestCase e aplica
a métrica ExecutionAccuracy. Deve rodar com o MESMO procedimento para baseline e
fine-tuned, garantindo comparabilidade.

Execute com:  pytest tests/
"""

import os
import json
import pytest
from deepeval.test_case import LLMTestCase
from custom_metrics.execution_accuracy import ExecutionAccuracy

# Ajuste conforme sua estrutura em data/ e results/
SPIDER_DB_ROOT = os.environ.get("SPIDER_DB_ROOT", "data/spider/database")
PREDICTIONS = os.environ.get("PREDICTIONS", "results/baseline/predictions.jsonl")


def load_test_cases(predictions_path: str):
    """lê predictions_path e devolver uma lista de LLMTestCase com:
        - input            = pergunta (+ esquema)
        - actual_output    = saída bruta do modelo
        - expected_output  = SQL de referência (gold)
        - additional_metadata = {"db_id": ...}
    """
    lista = []
    with open(predictions_path, 'r') as arq:
      for linha in arq:
        item = json.loads(linha)
        lista.append(LLMTestCase(
          input=item["input"],
          actual_output=item["actual_output"],
          expected_output=item["expected_output"],
          additional_metadata=item["db_id"]
        ))
    return lista

@pytest.mark.parametrize("test_case", load_test_cases(PREDICTIONS) if os.path.exists(PREDICTIONS) else [])
def test_execution_accuracy(test_case):
    metric = ExecutionAccuracy(db_root=SPIDER_DB_ROOT)
    score = metric.measure(test_case)
    assert score in (0.0, 1.0)
    # A agregação (média = Execution Accuracy do conjunto) deve ser reportada
    # separadamente; este teste valida a métrica caso a caso.
