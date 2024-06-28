from collections import Counter

def calculate_majority_vote(predictions):
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common

def tokenize_function(example,tokenizer):
    start_prompt = 'Summarize the following conversation. \n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True,
                                     return_tensors='pt').input_ids
    example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True,
                                 return_tensors='pt').input_ids

    return example