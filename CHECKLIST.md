# Checklist de aderência aos requisitos

Marque `[x]` e indique **onde** cada requisito foi cumprido (arquivo, função, seção do relatório).
A correção usa este checklist como guia — preencher com honestidade ajuda você e o corretor.

## Forma da entrega
- [x] Sem `.zip`/`.rar`, sem `venv/`, sem `__pycache__`, sem datasets/checkpoints versionados (`.gitignore` presente)
- [x] Estrutura na raiz (`scripts/`, `custom_metrics/`, `tests/`, `configs/`, `requirements.txt`, `README.md`)
- [x] `RELATORIO.pdf` na raiz, abre normalmente, texto selecionável, ≤ 10 páginas, formato IEEE/ACM
- [x] `requirements.txt` com versões fixadas
- [x] Nomes de arquivo sem espaços/acentos

## Fase 1 — Métrica Execution Accuracy
- [x] Classe herda de `deepeval.metrics.BaseMetric` — _onde:_ `custom_metrics/execution_accuracy.py`
- [x] `measure(self, test_case)` implementado
- [x] Extração robusta do SQL da saída bruta (remove markdown/texto explicativo)
- [x] Conexão com SQLite da base de teste do Spider
- [x] Execução em transação segura (try/except para erro de sintaxe)
- [x] Comparação **insensível à ordem**, **exceto** quando há `ORDER BY` (ordem respeitada)
- [x] Retorna 1.0 (sucesso) / 0.0 (falha)
- [x] Mesma métrica usada de forma idêntica no baseline e no fine-tuned

## Fase 2 — Baseline
- [x] Prompt few-shot com **esquema do banco + pergunta + 3 exemplos** (do train split)
- [x] Template de prompt fixo
- [x] Modelo base avaliado no Spider **dev** split
- [x] SQL gerado e resultado (sucesso/falha) registrados — _onde:_ `results/baseline/`

## Fase 3 — Fine-tuning
- [x] LoRA (ou QLoRA) implementado
- [x] Config documentada: r, alpha, dropout, target_modules — _onde:_ `configs/`, relatório
- [ ] **≥ 2 configurações** de hiperparâmetros testadas (ex.: learning_rate ou épocas)
- [x] Hardware (GPU/VRAM) documentado no relatório

## Fase 4 — Avaliação na tarefa-alvo
- [x] Métrica integrada em **pytest** — _onde:_ `tests/test_execution_accuracy.py`
- [x] Fine-tuned avaliado no Spider dev com o **mesmo procedimento** do baseline
- [ ] Resultados em `results/finetuned/`

## Fase 5 — Regressão MMLU
- [x] Suíte com **exatamente 150 questões** (50 STEM / 50 Humanidades / 50 Ciências Sociais)
- [x] Avaliação **5-shot**, com os **mesmos 5 exemplos** para base e fine-tuned
- [x] Acurácia **agregada e por categoria** — _onde:_ `results/mmlu/`
- [ ] Variação percentual base × fine-tuned calculada

## Reprodutibilidade
- [x] Seeds fixadas em todas as operações estocásticas
- [x] Geração determinística na avaliação (greedy, temperature = 0)
- [ ] Discussão de contaminação de dados no relatório

## Relatório
- [x] Metodologia: pipeline de dados, tabela de hiperparâmetros LoRA, arquitetura da métrica
- [ ] Resultados com acurácias + análise de erros (2-3 exemplos de falha do fine-tuned)
- [ ] Discussão do trade-off especialização × generalização
