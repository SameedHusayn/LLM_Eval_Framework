from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import torch
import numpy as np
import evaluate
from utils import calculate_majority_vote, generate_summaries
from data import load_dataset_subset
from datasets import load_dataset, DatasetDict


def evaluate_hellaswag(tokenizer, model, subset_size=10):
    dataset = load_dataset("hellaswag", trust_remote_code=True)
    dataset = dataset['validation'].select(range(subset_size))
    accuracy_metric = load_metric("accuracy")

    predictions_list = []
    references_list = []

    for example in dataset:
        context = str(example['ctx_a'])
        endings = example['endings']
        label_int = int(example['label'])

        all_predictions = []

        for _ in range(5):
            for ending in endings:
                inputs = tokenizer([context + " " + ending], return_tensors="pt", padding='max_length', max_length=64, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                mean_logits = logits[:, -1, :].mean(dim=-1)
                prediction = mean_logits.argmax().item()
                all_predictions.append(prediction)

        final_prediction = calculate_majority_vote(all_predictions)
        predictions_list.append(final_prediction)
        references_list.append(label_int)
        accuracy_metric.add_batch(predictions=[final_prediction], references=[label_int])

    accuracy_score = accuracy_metric.compute()
    print(f"HellaSwag Accuracy: {accuracy_score['accuracy']}")

def evaluate_glue_cola(tokenizer, model, subset_size=100):
    dataset = load_dataset("glue", "cola", split='validation')
    validation_subset = dataset.select(100)

    predictions_list = []
    references_list = []

    for example in validation_subset:
        sentence = str(example['sentence'])
        label_int = int(example['label'])

        all_predictions = []

        for _ in range(5):
            inputs = tokenizer([sentence], return_tensors="pt", padding='max_length', max_length=64, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            mean_logits = logits[:, -1, :].mean(dim=-1)
            prediction = mean_logits.argmax().item()
            all_predictions.append(prediction)

        final_prediction = calculate_majority_vote(all_predictions)
        predictions_list.append(final_prediction)
        references_list.append(label_int)

    accuracy = accuracy_score(references_list, predictions_list)
    print(f"GLUE CoLA Accuracy: {accuracy}")

def evaluate_glue_sst2(tokenizer, model, subset_size=100):
    dataset = load_dataset("glue", "sst2", split='validation')
    dataset = dataset.select(range(subset_size))
    predictions_list = []
    references_list = []

    for example in dataset:
        sentence = str(example['sentence'])
        label_int = int(example['label'])

        all_predictions = []

        for _ in range(5):
            inputs = tokenizer([sentence], return_tensors="pt", padding='max_length', max_length=64, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            mean_logits = logits[:, -1, :].mean(dim=-1)
            prediction = mean_logits.argmax().item()
            all_predictions.append(prediction)

        final_prediction = calculate_majority_vote(all_predictions)
        predictions_list.append(final_prediction)
        references_list.append(label_int)

    accuracy = accuracy_score(references_list, predictions_list)
    print(f"GLUE SST-2 Accuracy: {accuracy}")

def evaluate_glue_qqp(tokenizer, model, subset_size=100):
    dataset = load_dataset("glue", "qqp", split='validation')
    dataset = dataset.select(range(subset_size))

    predictions_list = []
    references_list = []

    for example in dataset:
        question1 = str(example['question1'])
        question2 = str(example['question2'])
        label_int = int(example['label'])

        all_predictions = []

        for _ in range(5):
            inputs = tokenizer([question1, question2], return_tensors="pt", padding='max_length', max_length=128, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            mean_logits = logits[:, -1, :].mean(dim=-1)
            prediction = mean_logits.argmax().item()
            all_predictions.append(prediction)

        final_prediction = calculate_majority_vote(all_predictions)
        predictions_list.append(final_prediction)
        references_list.append(label_int)

    accuracy = accuracy_score(references_list, predictions_list)
    precision, recall, f1, _ = precision_recall_fscore_support(references_list, predictions_list, average='binary')

    print(f"GLUE QQP Accuracy: {accuracy}")
    print(f"GLUE QQP Precision: {precision}")
    print(f"GLUE QQP Recall: {recall}")
    print(f"GLUE QQP F1: {f1}")

def evaluate_glue_stsb(tokenizer, model, subset_size=100):
    dataset = load_dataset("glue", "stsb", split='validation')
    dataset = dataset.select(range(subset_size))

    predictions_list = []
    references_list = []

    for example in dataset:
        sentence1 = str(example['sentence1'])
        sentence2 = str(example['sentence2'])
        label_float = float(example['label'])

        all_predictions = []

        for _ in range(5):
            inputs = tokenizer([sentence1, sentence2], return_tensors="pt", padding='max_length', max_length=128, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            mean_logits = logits[:, -1, :].mean(dim=-1)
            prediction = mean_logits.argmax().item()
            all_predictions.append(prediction)

        final_prediction = np.mean(all_predictions)
        predictions_list.append(final_prediction)
        references_list.append(label_float)

    pearson_corr, _ = pearsonr(references_list, predictions_list)
    spearman_corr, _ = spearmanr(references_list, predictions_list)

    print(f"GLUE STS-B Pearson Correlation: {pearson_corr}")
    print(f"GLUE STS-B Spearman Correlation: {spearman_corr}")

def evaluate_dialogsum(tokenizer, model, subset_size=10):
    dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(dataset_name)
    reduced_test_set = dataset['test'].select(range(subset_size))
    dialogues = reduced_test_set['dialogue']
    human_baseline_summaries = reduced_test_set['summary']

    generated_summaries = generate_summaries(tokenizer, model, dialogues)

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')

    rouge_results = rouge.compute(predictions=generated_summaries, references=human_baseline_summaries)
    bleu_results = bleu.compute(predictions=generated_summaries, references=human_baseline_summaries)
    meteor_results = meteor.compute(predictions=generated_summaries, references=human_baseline_summaries)

    print(f"DialogSum ROUGE: {rouge_results}")
    print(f"DialogSum BLEU: {bleu_results}")
    print(f"DialogSum METEOR: {meteor_results}")

def evaluate_perplexity(tokenizer, model):
    from datasets import load_dataset
    import torch
    import math

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

        log_likelihood = outputs.loss * trg_len
        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print(f"Perplexity: {ppl.item()}")
