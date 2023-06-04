# reddit-llama

This repository contains code for fine-tuning the [LLaMA](https://arxiv.org/pdf/2302.13971.pdf) model using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf) on any task provided some training input-response pairs. The training of the 7B model is possible with as little as 16GB of GPU memory.


We trained the model on dataset containing question-answer pairs collected from the AskReddit subreddit. This dataset can be found on [Kaggle](https://www.kaggle.com/datasets/vaclavhalama/reddit-questions-and-answers). Only a subset of this dataset (``reddit60k.jsonl``) was used to train the model. The adapter weights are provided in ``askreddit_v1``.


### Fine-tuning

- The hyper-parameters can be changed in ``config.py``.
- The training dataset must be in JSON-Lines format with one JSON object per line with one key-value pair corresponding to the input and the second to the desired response. These keys needs to also be set in the config as *src_column* and *tgt_column*.
- The fine-tuning  is initiated by running the ``finetune.py`` script.


### Demo

Demo usage of the fine-tuned model is shown in [demo.ipynb](https://github.com/vaclav-h/reddit-llama/blob/main/demo.ipynb).
