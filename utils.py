def prepare_prompt(src):
    prompt = "### Input:\n{}\n\n### Response:".format(src)
    return prompt

def preprocess(example, src_column, tgt_column, tokenizer, max_length=128):
    
    # Prepare prompt
    src = prepare_prompt(example[src_column])
    
    # Append end-of-sequence token to target
    tgt = f"{example[tgt_column]}{tokenizer.eos_token}"
    
    # Construct sample
    sample = src + tgt
    
    # Tokenize sample and get length of the tokenized prompt
    sample_tokenized = tokenizer(sample, max_length=max_length,
                                 truncation=True, padding='max_length',
                                 return_tensors='pt')
    src_tokenized = tokenizer(src, max_length=max_length,
                              truncation=True, padding='max_length',
                              return_tensors='pt')
    src_len = src_tokenized['input_ids'][0].ne(tokenizer.pad_token_id).sum().item()
    
    # Prepare input ids
    input_ids = sample_tokenized['input_ids'][0]
    
    # Prepare labels
    # put -100 instead of the first src_len tokens and padding tokens in order to not compute loss there
    labels = input_ids.detach().clone()
    labels[:src_len] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Prepare attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    
    return {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "labels" : labels
    }