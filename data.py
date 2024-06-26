from datasets import load_dataset

def load_dataset_subset(dataset_name, subset_size, split="validation"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset.select(range(subset_size))
