from evaluations import evaluate_hellaswag, evaluate_glue_cola, evaluate_glue_sst2, evaluate_glue_qqp, evaluate_glue_stsb, evaluate_dialogsum, evaluate_perplexity, evaluate_mmlu
# Model choices and their descriptions
models = {
    '1': {
        'name': "microsoft/Phi-3-mini-4k-instruct",
        'description': "1. Phi-3-Mini-4K-Instruct"
    },
    '2': {
        'name': "mistralai/Mistral-7B-v0.1",
        'description': "2. Mistral-7B-v0.1"
    },
    '3': {
        'name': "meta-llama/Llama-2-7b-hf",
        'description': "3. Llama-2-7b-hf"
    },
    '4': {
        'name': "meta-llama/Meta-Llama-3-8B",
        'description': "4. Meta-Llama-3-8B"
    },
    '5': {
        'name': "google/gemma-2-9b",
        'description': "5. Gemma-2-9b"
    },
    '6': {
        'name': "Qwen/Qwen-7B",
        'description': "6. wen-7B"
    }
}

# Task choices and their descriptions
tasks = {
    '1': {
        'function': evaluate_hellaswag,
        'description': "1. Hellaswag"
    },
    '2': {
        'function': evaluate_glue_cola,
        'description': "2. COLA"
    },
    '3': {
        'function': evaluate_glue_sst2,
        'description': "3. SST-2"
    },
    '4': {
        'function': evaluate_glue_qqp,
        'description': "4. QQP"
    },
    '5': {
        'function': evaluate_glue_stsb,
        'description': "5. STSB"
    },
    '6': {
        'function': evaluate_dialogsum,
        'description': "6. DialogSum"
    },
    '7': {
        'function': evaluate_perplexity,
        'description': "7. Perplexity"
    },
    '8': {
        'function': evaluate_mmlu,
        'description': "8. MMLU"
    }
}
