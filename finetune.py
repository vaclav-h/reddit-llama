import torch
import os
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from config import get_config
from utils import preprocess
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training
)


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



def train(cfg):
    use_wandb = len(cfg["wandb_project"]) > 0
    if use_wandb:
        os.environ['WANDB_PROJECT'] = cfg['wandb_project']
    
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    num_new_tokens = tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN,
                                                   "eos_token": DEFAULT_EOS_TOKEN,
                                                   "bos_token": DEFAULT_BOS_TOKEN,
                                                   "unk_token": DEFAULT_UNK_TOKEN})
    
    model = LlamaForCausalLM.from_pretrained(cfg["checkpoint"],
                                             load_in_8bit=True,
                                             torch_dtype=torch.float16,
                                             device_map='auto')
    
    # We added pad_token so we need to resize the models' input and output embeddings
    # New embeddings will be initialized as the average of the remaining embeddings
    model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    train_ds = load_dataset("json", data_files=cfg['train_data'])['train']
    train_ds = train_ds.map(lambda example: preprocess(example,
                                                       src_column=cfg["src_column"],
                                                       tgt_column=cfg["tgt_column"],
                                                       tokenizer=tokenizer,
                                                       max_length=cfg["max_input_length"]),
                            remove_columns=train_ds.column_names,
                            num_proc=8)
    train_ds.set_format("pt", columns=["input_ids", "labels", "attention_mask"])
    
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    
    args = TrainingArguments(
                output_dir=cfg['output_dir'],
                num_train_epochs=cfg['epochs'],
                per_device_train_batch_size=cfg['batch_size'],
                gradient_accumulation_steps=cfg['gradient_accumulations'],
                learning_rate=cfg['lr'],
                optim='adamw_torch',
                fp16=True,
                logging_steps=10,
                save_strategy='no',
                report_to="wandb" if use_wandb else None,
                run_name=cfg["wandb_run"] if use_wandb else None
            )
    
    trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    train_dataset=train_ds
                )
    
    trainer.train()
    
    model.save_pretrained(cfg['output_dir'])
    tokenizer.save_pretrained(cfg['output_dir'])

if __name__ == '__main__':
    cfg = get_config()
    train(cfg)
