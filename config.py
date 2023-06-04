def get_config():
    return {
        "checkpoint" : "decapoda-research/llama-7b-hf",
        "train_data" : "reddit60k.jsonl",
        "src_column" : "title",
        "tgt_column" : "comment",
        "max_input_length" : 128,
        "output_dir" : "askreddit_v1",
        "epochs" : 5,
        "batch_size" : 32, # Effective batch size = batch_size * gradient_accumulations * number of GPUs
        "gradient_accumulations" : 4,
        "lr" : 3e-4,
        "wandb_project" : "",
        "wandb_run" : ""
    }
