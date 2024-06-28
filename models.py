from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainingArguments, pipeline, Trainer

def init_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, trust_remote_code=True)
    return tokenizer, model
