from models import init_model_and_tokenizer
from data import load_dataset_subset
from evaluations import evaluate_hellaswag, evaluate_glue_mrpc, evaluate_glue_cola, evaluate_glue_sst2, evaluate_glue_qqp, evaluate_glue_stsb, evaluate_dialogsum, evaluate_perplexity, evaluate_mmlu
from huggingface_hub import notebook_login

import os, shutil
def main():
    notebook_login()
    
    print("Select models:")
    print("1. microsoft/Phi-3-mini-4k-instruct")
    print("2. mistralai/Mistral-7B-v0.1")
    print("3. meta-llama/Llama-2-7b-hf")
    print("4. meta-llama/Meta-Llama-3-8B")
    print("5. google/gemma-2-9b")
    print("6. Qwen/Qwen-7B")

    model_choices = input("Enter the numbers of the models you want to evaluate, separated by commas: ").strip().split(',')

    print("Select evaluation tasks:")
    print("1. Hellaswag")
    print("2. COLA")
    print("3. SST-2")
    print("4. QQP")
    print("5. STSB")
    print("6. DialogSum")
    print("7. Perplexity")
    print("8. MMLU")

    task_choices = input("Enter the numbers of the tasks you want to evaluate, separated by commas: ").strip().split(',')

    models = {
        '1': "microsoft/Phi-3-mini-4k-instruct",
        '2': "mistralai/Mistral-7B-v0.1",
        '3': "meta-llama/Llama-2-7b-hf",
        '4': "meta-llama/Meta-Llama-3-8B",
        '5': "google/gemma-2-9b",
        '6': "Qwen/Qwen-7B"
          }

    selected_models = [models[model.strip()] for model in model_choices if model.strip() in models]
    tasks = {
        '1': evaluate_hellaswag,
        '2': evaluate_glue_cola,
        '3': evaluate_glue_sst2,
        '4': evaluate_glue_qqp,
        '5': evaluate_glue_stsb,
        '6': evaluate_dialogsum,
        '7': evaluate_perplexity,
        '8': evaluate_mmlu }


    selected_tasks = [tasks[task.strip()] for task in task_choices if task.strip() in tasks]

    for model_name in selected_models:
      # Path to the model directory, replace 'model_name' with the actual model name
      model_dir = os.path.join('/root/.cache/huggingface/transformers/models--',model_name)
      # Check if the directory exists and remove it
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
          print(f"Deleted model directory: {model_dir}")
      else:
          print(f"Model directory does not exist: {model_dir}")
      tokenizer, model = init_model_and_tokenizer(model_name)
      for task_name in selected_tasks:
        task_name(tokenizer, model)
      print(model, "Evaluated!")


if __name__ == "__main__":
    main()