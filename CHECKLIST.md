# Checklist de aderência aos requisitos

Marque `[x]` e indique **onde** cada requisito foi cumprido (arquivo, função, seção do relatório).
A correção usa este checklist como guia — preencher com honestidade ajuda você e o corretor.

## Forma da entrega
- [ ] Sem `.zip`/`.rar`, sem `venv/`, sem `__pycache__`, sem datasets/checkpoints versionados (`.gitignore` presente)
- [ ] Estrutura na raiz (`scripts/`, `custom_metrics/`, `tests/`, `configs/`, `requirements.txt`, `README.md`)
- [ ] `RELATORIO.pdf` na raiz, abre normalmente, texto selecionável, ≤ 10 páginas, formato IEEE/ACM
- [ ] `requirements.txt` com versões fixadas
- [ ] Nomes de arquivo sem espaços/acentos

## Fase 1 — Métrica Execution Accuracy
- [ ] Classe herda de `deepeval.metrics.BaseMetric` — _onde:_ `custom_metrics/execution_accuracy.py`
- [ ] `measure(self, test_case)` implementado
- [ ] Extração robusta do SQL da saída bruta (remove markdown/texto explicativo)
- [ ] Conexão com SQLite da base de teste do Spider
- [ ] Execução em transação segura (try/except para erro de sintaxe)
- [ ] Comparação **insensível à ordem**, **exceto** quando há `ORDER BY` (ordem respeitada)
- [ ] Retorna 1.0 (sucesso) / 0.0 (falha)
- [ ] Mesma métrica usada de forma idêntica no baseline e no fine-tuned

## Fase 2 — Baseline
- [ ] Prompt few-shot com **esquema do banco + pergunta + 3 exemplos** (do train split)
- [ ] Template de prompt fixo
- [ ] Modelo base avaliado no Spider **dev** split
- [ ] SQL gerado e resultado (sucesso/falha) registrados — _onde:_ `results/baseline/`

## Fase 3 — Fine-tuning
- [ ] LoRA (ou QLoRA) implementado
- [ ] Config documentada: r, alpha, dropout, target_modules — _onde:_ `configs/`, relatório
- [ ] **≥ 2 configurações** de hiperparâmetros testadas (ex.: learning_rate ou épocas)
- [ ] Hardware (GPU/VRAM) documentado no relatório

## Fase 4 — Avaliação na tarefa-alvo
- [ ] Métrica integrada em **pytest** — _onde:_ `tests/test_execution_accuracy.py`
- [ ] Fine-tuned avaliado no Spider dev com o **mesmo procedimento** do baseline
- [ ] Resultados em `results/finetuned/`

## Fase 5 — Regressão MMLU
- [ ] Suíte com **exatamente 150 questões** (50 STEM / 50 Humanidades / 50 Ciências Sociais)
- [ ] Avaliação **5-shot**, com os **mesmos 5 exemplos** para base e fine-tuned
- [ ] Acurácia **agregada e por categoria** — _onde:_ `results/mmlu/`
- [ ] Variação percentual base × fine-tuned calculada

## Reprodutibilidade
- [ ] Seeds fixadas em todas as operações estocásticas
- [ ] Geração determinística na avaliação (greedy, temperature = 0)
- [ ] Discussão de contaminação de dados no relatório

## Relatório
- [ ] Metodologia: pipeline de dados, tabela de hiperparâmetros LoRA, arquitetura da métrica
- [ ] Resultados com acurácias + análise de erros (2-3 exemplos de falha do fine-tuned)
- [ ] Discussão do trade-off especialização × generalização
