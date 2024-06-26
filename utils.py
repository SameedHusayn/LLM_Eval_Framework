from collections import Counter

def calculate_majority_vote(predictions):
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common

def generate_summaries(tokenizer, model, dialogues):
    generated_summaries = []

    for dialogue in dialogues:
        inputs = tokenizer.encode(dialogue, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_summaries.append(summary)

    return generated_summaries
