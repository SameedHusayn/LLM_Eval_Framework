from models import init_model_and_tokenizer
from huggingface_hub import notebook_login
from constants import models, tasks

import os
import shutil

def main():
    """
    Main function to select models and evaluation tasks, initialize the models and tokenizers, and evaluate the selected models on the selected tasks.

    The function performs the following steps:
    1. Logs in to the notebook environment.
    2. Displays available models and prompts the user to select models for evaluation.
    3. Displays available evaluation tasks and prompts the user to select tasks for evaluation.
    4. Initializes the selected models and tokenizers.
    5. Evaluates the selected models on the selected tasks.
    6. Deletes the model directories after evaluation if they exist.

    Args:
        None

    Returns:
        None
    """

    notebook_login()
    
    print("Select models:")
    for model in models.values():
        print(model['description'])

    model_choices = input("Enter the numbers of the models you want to evaluate, separated by commas: ").strip().split(',')

    print("Select evaluation tasks:")
    for task in tasks.values():
        print(task['description'])

    task_choices = input("Enter the numbers of the tasks you want to evaluate, separated by commas: ").strip().split(',')

    selected_models = [models[model.strip()]['name'] for model in model_choices if model.strip() in models]
    selected_tasks = [tasks[task.strip()]['function'] for task in task_choices if task.strip() in tasks]

    for model_name in selected_models:
        # Path to the model directory, replace 'model_name' with the actual model name
        model_dir = os.path.join('/root/.cache/huggingface/transformers/models--', model_name)
        # Check if the directory exists and remove it
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"Deleted model directory: {model_dir}")
        else:
            print(f"Model directory does not exist: {model_dir}")
        tokenizer, model = init_model_and_tokenizer(model_name)
        for task in selected_tasks:
            task(tokenizer, model)
        print(f"{model_name} evaluated!")

if __name__ == "__main__":
    main()
