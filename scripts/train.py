"""Fine-tuning com LoRA/QLoRA (Hugging Face TRL + PEFT).

Roda UMA configuração de hiperparâmetros por vez, lida de um arquivo em configs/.
Execute para CADA config (mínimo 2) — Fase 3.3.

    python scripts/train.py --config configs/lora_config_a.yaml

FIXE as seeds (set_seed) para reprodutibilidade. Documente o hardware (GPU/VRAM) no relatório.
"""

import argparse
import torch
import json
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from peft import LoraConfig


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="caminho do YAML em configs/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    #  1
    set_seed(cfg["seed"])

    #  2. carregar modelo base (4-bit se QLoRA) + tokenizer
    quantization_config = BitsAndBytesConfig(load_in_4bit=cfg["model"]["load_in_4bit"], bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
      cfg["model"]["base_model"],
      dtype=torch.float16,
      device_map="auto",
      quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    if (tokenizer.pad_token == None):
        tokenizer.pad_token = tokenizer.eos_token

    #  3. montar LoraConfig (r, lora_alpha, lora_dropout, target_modules) a partir de cfg
    lora_cfg = cfg["lora"]
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"]
    )

    #  4. carregar data/processed/train_chat.jsonl
    data = []
    with open("data/processed/train_chat.jsonl", 'r') as arq:
        for linha in arq:
            meu_json = json.loads(linha)
            data.append({"messages": [{"role": "system", "content": "Dado um esquema e uma pergunta, escreva uma consulta SQL que a responde."}, {"role": "user", "content": meu_json["user"]}, {"role": "assistant", "content": meu_json["assistant"]}]})
    train_dataset = Dataset.from_list(data)

    #  5. SFTTrainer / TRL com os hiperparâmetros de cfg (learning_rate, epochs, ...)
    train_cfg = cfg["train"]
    train_args = SFTConfig(
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        output_dir=train_cfg["output_dir"],
        max_length=train_cfg["max_seq_length"]
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=train_args
    )
    trainer.train()

    #  6. salvar o adapter em output_dir (NÃO versionar)
    trainer.save_model(train_cfg["output_dir"])
    print(f"Treinamento concluído! Adapter salvo em: {train_cfg['output_dir']}")

if __name__ == "__main__":
    main()
