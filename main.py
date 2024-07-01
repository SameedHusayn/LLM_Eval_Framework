from models import init_model_and_tokenizer
from data import load_dataset_subset
from evaluations import evaluate_hellaswag, evaluate_glue_cola, evaluate_glue_sst2, evaluate_glue_qqp, evaluate_glue_stsb, evaluate_dialogsum, evaluate_perplexity, evaluate_mmlu

def print_task_choices():
    print("Select evaluation tasks:")
    print("1. Hellaswag")
    print("2. COLA")
    print("3. SST-2")
    print("4. QQP")
    print("5. STSB")
    print("6. DialogSum")
    print("7. Perplexity")
    print("8. MMLU")
    print("Enter task numbers separated by commas (e.g., 1, 3, 6):")

def main():
    print_task_choices()
    task_choices = input("Enter the numbers of the tasks you want to evaluate, separated by commas: ").strip().split(',')

    tasks = {
        '1': evaluate_hellaswag,
        '2': evaluate_glue_cola,
        '3': evaluate_glue_sst2,
        '4': evaluate_glue_qqp,
        '5': evaluate_glue_stsb,
        '6': evaluate_dialogsum,
        '7': evaluate_perplexity,
        '8': evaluate_mmlu
    }

    selected_tasks = [tasks[task.strip()] for task in task_choices if task.strip() in tasks]
    
    if selected_tasks:
        model_name = input("Enter the model name or path (default: microsoft/Phi-3-mini-4k-instruct): ") or "microsoft/Phi-3-mini-4k-instruct"
        # tokenizer, model = init_model_and_tokenizer(model_name)
        tokenizer = "tokenzier"
        model = "model"
        
        for task_func in selected_tasks:
            task_func(tokenizer, model)
    else:
        print("No valid tasks selected. Please enter valid task numbers.")

if __name__ == "__main__":
    main()
