"""Fine-tuning com LoRA/QLoRA (Hugging Face TRL + PEFT).

Roda UMA configuração de hiperparâmetros por vez, lida de um arquivo em configs/.
Execute para CADA config (mínimo 2) — Fase 3.3.

    python scripts/train.py --config configs/lora_config_a.yaml

FIXE as seeds (set_seed) para reprodutibilidade. Documente o hardware (GPU/VRAM) no relatório.
"""

import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="caminho do YAML em configs/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    # TODO:
    #  1. set_seed(cfg["seed"])
    #  2. carregar modelo base (4-bit se QLoRA) + tokenizer
    #  3. montar LoraConfig (r, lora_alpha, lora_dropout, target_modules) a partir de cfg
    #  4. carregar data/processed/train_chat.jsonl
    #  5. SFTTrainer / TRL com os hiperparâmetros de cfg (learning_rate, epochs, ...)
    #  6. salvar o adapter em output_dir (NÃO versionar)
    raise NotImplementedError


if __name__ == "__main__":
    main()
