from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainingArguments, pipeline, Trainer, BertTokenizer, BertForSequenceClassification

def init_model_and_tokenizer(model_name):
  """
    Initializes and returns the tokenizer and model for the given model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the initialized tokenizer and model.
  """

  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  if model_name == 'Qwen/Qwen-7B':
    tokenizer.eos_token_id = 151646
    tokenizer.pad_token_id = 151645
    tokenizer.bos_token_id = 151648
  else:
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
  # model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
  return tokenizer, model
