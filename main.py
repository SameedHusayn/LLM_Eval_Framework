from models import init_model_and_tokenizer
from data import load_dataset_subset
from evaluations import evaluate_hellaswag, evaluate_glue_cola, evaluate_glue_sst2, evaluate_glue_qqp, evaluate_glue_stsb, evaluate_dialogsum, evaluate_perplexity, evaluate_mmlu
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on various benchmarks")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Model name or path")
    parser.add_argument("--task", type=str, required=True, choices=["hellaswag", "cola", "sst2", "qqp", "stsb", "dialogsum", "perplexity", "mmlu"], help="Evaluation task")
    args = parser.parse_args()

    tokenizer, model = init_model_and_tokenizer(args.model_name)

    if args.task == "hellaswag":
        evaluate_hellaswag(tokenizer, model)
    elif args.task == "cola":
        evaluate_glue_cola(tokenizer, model)
    elif args.task == "sst2":
        evaluate_glue_sst2(tokenizer, model)
    elif args.task == "qqp":
        evaluate_glue_qqp(tokenizer, model)
    elif args.task == "stsb":
        evaluate_glue_stsb(tokenizer, model)
    elif args.task == "dialogsum":
        evaluate_dialogsum(tokenizer, model)
    elif args.task == "perplexity":
        evaluate_perplexity(tokenizer, model)
    elif args.task == "mmlu":
        evaluate_mmlu(tokenizer, model)

if __name__ == "__main__":
    main()