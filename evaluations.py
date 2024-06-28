from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import torch
import numpy as np
import evaluate
from utils import calculate_majority_vote, tokenize_function, format_subject, format_example, gen_prompt
from data import load_dataset_subset
from datasets import load_dataset, DatasetDict
from transformers import GenerationConfig
from tqdm import tqdm
import csv
import pandas as pd


def evaluate_hellaswag(tokenizer, model, subset_size=10):
    dataset = load_dataset("hellaswag", trust_remote_code=True)
    dataset = dataset['validation'].select(range(subset_size))
    accuracy_metric = load_metric("accuracy", trust_remote_code=True)

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
    reduced_test_set = dataset['test'].select(range(10))

    # Updating the dataset with the reduced test set
    dataset = DatasetDict({
        "train": dataset['train'],
        "validation": dataset['validation'],
        "test": reduced_test_set
    })

    tokenize_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenize_datasets = tokenize_datasets.remove_columns(['id', 'topic', 'dialogue',
                                                     'summary'])

    # Load evaluation metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')

    # Get dialogues and human summaries
    dialogues = dataset['test']['dialogue']
    human_baseline_summaries = dataset['test']['summary']

    # Initialize lists for generated summaries
    original_model_summaries = []

    # Generate summaries
    for dialogue in dialogues:
        prompt = f"Summarize the following conversations.\n\n{dialogue}\n\nSummary: "
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        original_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=100, num_beams=1))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        original_model_summaries.append(original_model_text_output)

    # Evaluate using ROUGE
    rouge_results = rouge.compute(predictions=original_model_summaries, references=human_baseline_summaries)
    print(f'ROUGE Results: \n{rouge_results}\n')

    # Evaluate using BLEU
    bleu_results = bleu.compute(predictions=original_model_summaries, references=human_baseline_summaries)
    print(f'BLEU Results: \n{bleu_results}\n')

    # Evaluate using METEOR
    meteor_results = meteor.compute(predictions=original_model_summaries, references=human_baseline_summaries)
    print(f'METEOR Results: \n{meteor_results}\n')

def evaluate_perplexity(tokenizer, model):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt", truncation=True, max_length=4096)
    max_length = model.config.max_length
    stride = 512
    seq_len = encodings.input_ids.size(1)
    device = "cuda"

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")

def calculate_mmlu(subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    choices = ["A", "B", "C", "D"]

    for i in range(test_df.shape[0]):
        k = dev_df.shape[0]
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
#         print(prompt)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
        next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)
#         print(logits)

        next_token_logits = next_token_logits.flatten()
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
        tokens_of_interest = [
            tokenizer("A", add_special_tokens=False).input_ids[-1],
            tokenizer("B", add_special_tokens=False).input_ids[-1],
            tokenizer("C", add_special_tokens=False).input_ids[-1],
            tokenizer("D", add_special_tokens=False).input_ids[-1],
        ]
        probs = next_token_probs[tokens_of_interest].tolist()
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
#         print(pred, choices[label])
        cor = pred == choices[label]
        cors.append(cor)
#         print(np.sum(cors)/len(cors))
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def evaluate_mmlu(tokenizer, model):
    dataset = load_dataset("cais/mmlu", 'abstract_algebra','anatomy', trust_remote_code=True)
    test = pd.DataFrame(dataset['test'])
    dev = pd.DataFrame(dataset['dev'])
    results = {}

    subjects = sorted(dev['subject'].value_counts().keys())
    for subject in subjects:
        cor, acc, prob = calculate_mmlu(subject, model, tokenizer, dev[dev['subject'] == subject], test[test['subject'] == subject])
        # print(cor, acc, prob)
        results[subject] = acc

    avg_accuracy = np.mean(list(results.values()))
    print("Average accuracy across all subjects: {:.3f}".format(avg_accuracy))
    results["Average Accuracy"] = avg_accuracy
