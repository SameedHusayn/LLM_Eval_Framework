# LLM Evaluation Framework

This repository contains a comprehensive framework for evaluating large language models (LLMs) using various datasets and tasks. The framework includes scripts for model initialization, dataset loading, and evaluation metrics calculation across different tasks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the Evaluation](#running-the-evaluation)
- [Scripts Overview](#scripts-overview)
  - [evaluations.py](#evaluationspy)
  - [main.py](#mainpy)
  - [models.py](#modelspy)
  - [utils.py](#utilspy)
- [Supported Models](#supported-models)
- [Supported Tasks](#supported-tasks)
- [Requirements](#requirements)
- [Contributing](#contributing)
  
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LLM-Evaluation-Framework.git
    cd LLM-Evaluation-Framework
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Evaluation

1. Run the `main.py` script to start the evaluation process:
    ```bash
    python main.py
    ```

2. Follow the prompts to select the models and tasks you want to evaluate.

## Scripts Overview

### evaluations.py

This script contains functions for evaluating various datasets and tasks. Each evaluation function computes metrics such as accuracy, precision, recall, F1 score, Pearson correlation, Spearman correlation, ROUGE, BLEU, METEOR, and perplexity.

#### Functions:

- `evaluate_hellaswag(tokenizer, model)`
- `evaluate_glue_cola(tokenizer, model)`
- `evaluate_glue_sst2(tokenizer, model)`
- `evaluate_glue_mrpc(tokenizer, model)`
- `evaluate_glue_qqp(tokenizer, model)`
- `evaluate_glue_stsb(tokenizer, model)`
- `evaluate_dialogsum(tokenizer, model)`
- `evaluate_perplexity(tokenizer, model)`
- `calculate_mmlu(model, tokenizer)`
- `evaluate_mmlu(tokenizer, model)`

### main.py

The entry point of the framework. It allows users to select models and tasks for evaluation. It also handles model initialization and dataset loading.

#### Key Functions:

- `main()`: Handles user input for selecting models and tasks, initializes models, and runs the evaluation functions.

### models.py

This script contains the function for initializing models and tokenizers.

#### Functions:

- `init_model_and_tokenizer(model_name)`: Initializes and returns the tokenizer and model for a given model name.

### utils.py

Contains utility functions for processing data and calculating metrics.

#### Functions:

- `calculate_majority_vote(predictions)`
- `tokenize_function(example, tokenizer, max_length)`
- `format_subject(subject)`
- `format_example(df, idx, include_answer=True)`
- `gen_prompt(train_df, subject, k=-1)`

## Supported Models

- microsoft/Phi-3-mini-4k-instruct
- mistralai/Mistral-7B-v0.1
- meta-llama/Llama-2-7b-hf
- meta-llama/Meta-Llama-3-8B
- google/gemma-2-9b
- Qwen/Qwen-7B

## Supported Tasks

- Hellaswag
- GLUE CoLA
- GLUE SST-2
- GLUE MRPC
- GLUE QQP
- GLUE STS-B
- DialogSum
- Perplexity (WikiText-2-Raw)
- MMLU (Massive Multitask Language Understanding)

## Requirements

- datasets==2.20.0
- evaluate==0.4.2
- rouge_score==0.1.2
- accelerate==0.31.0
- bitsandbytes==0.43.1
- scipy==1.14.0
- transformers
- torch==2.3.1
- scikit-learn==1.5.0
- tqdm==4.66.4
- pandas==2.2.1
- numpy==1.26.4
- inquirer==3.3.0
- tiktoken==0.7.0
- einops==0.8.0 
- transformers_stream_generator==0.0.5

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.
