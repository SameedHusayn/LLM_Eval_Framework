from collections import Counter

def calculate_majority_vote(predictions):
    """
    Calculates the majority vote from a list of predictions.

    Args:
        predictions (list): A list of predictions.

    Returns:
        The most common prediction from the list.
    """

    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common

def tokenize_function(example, tokenizer, max_length):
    """
    Tokenizes the input example for a summarization task.

    Args:
        example (dict): A dictionary containing 'dialogue' and 'summary' keys.
        tokenizer: The tokenizer to process the text data.
        max_length (int): The maximum length for the tokenized sequences.

    Returns:
        dict: The tokenized example with 'input_ids', 'attention_mask', and 'labels' keys.
    """

    start_prompt = 'Summarize the following conversation. \n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    tokenized_prompt = tokenizer(prompt, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    tokenized_summary = tokenizer(example['summary'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    example['input_ids'] = tokenized_prompt['input_ids']
    example['attention_mask'] = tokenized_prompt['attention_mask']
    example['labels'] = tokenized_summary['input_ids']
    return example


def format_subject(subject):
    """
    Formats a subject name by replacing underscores with spaces.

    Args:
        subject (str): The subject name with underscores.

    Returns:
        str: The formatted subject name with spaces.
    """

    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    """
    Formats a single example from a DataFrame for prompt generation.

    Args:
        df (DataFrame): The DataFrame containing the examples.
        idx (int): The index of the example to format.
        include_answer (bool): Whether to include the answer in the formatted example. Default is True.

    Returns:
        str: The formatted example as a string.
    """

    choices = ["A", "B", "C", "D"]
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}".format(df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.iloc[idx, k + 1]])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    """
    Generates a prompt from the training DataFrame for a given subject.

    Args:
        train_df (DataFrame): The DataFrame containing the training examples.
        subject (str): The subject of the questions.
        k (int): The number of examples to include in the prompt. Default is -1, which includes all examples.

    Returns:
        str: The generated prompt as a string.
    """

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
