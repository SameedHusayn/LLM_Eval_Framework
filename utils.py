from collections import Counter

def calculate_majority_vote(predictions):
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common

def tokenize_function(example, tokenizer, max_length):
    start_prompt = 'Summarize the following conversation. \n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    tokenized_prompt = tokenizer(prompt, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    tokenized_summary = tokenizer(example['summary'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    example['input_ids'] = tokenized_prompt['input_ids']
    example['attention_mask'] = tokenized_prompt['attention_mask']
    example['labels'] = tokenized_summary['input_ids']
    return example


choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}".format(df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.iloc[idx, k + 1]])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
