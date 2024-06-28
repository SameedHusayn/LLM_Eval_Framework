import inquirer
from models import init_model_and_tokenizer
from data import load_dataset_subset
from evaluations import evaluate_hellaswag, evaluate_glue_cola, evaluate_glue_sst2, evaluate_glue_qqp, evaluate_glue_stsb, evaluate_dialogsum, evaluate_perplexity, evaluate_mmlu

def main():
    questions = [
        inquirer.List(
            'task',
            message="Select an evaluation task",
            choices=["hellaswag", "cola", "sst2", "qqp", "stsb", "dialogsum", "perplexity", "mmlu"],
        ),
        inquirer.Text('model_name', message="Enter the model name or path", default="microsoft/Phi-3-mini-4k-instruct")
    ]
    
    answers = inquirer.prompt(questions)
    model_name = answers['model_name']
    task = answers['task']

    tokenizer, model = init_model_and_tokenizer(model_name)

    if task == "hellaswag":
        evaluate_hellaswag(tokenizer, model)
    elif task == "cola":
        evaluate_glue_cola(tokenizer, model)
    elif task == "sst2":
        evaluate_glue_sst2(tokenizer, model)
    elif task == "qqp":
        evaluate_glue_qqp(tokenizer, model)
    elif task == "stsb":
        evaluate_glue_stsb(tokenizer, model)
    elif task == "dialogsum":
        evaluate_dialogsum(tokenizer, model)
    elif task == "perplexity":
        evaluate_perplexity(tokenizer, model)
    elif task == "mmlu":
        evaluate_mmlu(tokenizer, model)

if __name__ == "__main__":
    main()
