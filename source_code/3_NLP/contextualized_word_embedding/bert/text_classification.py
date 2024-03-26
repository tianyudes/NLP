from datasets import load_dataset, load_metric
import random, datasets
import pandas as pd
from transformers import AutoTokenizer
    



def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    
    df.to_csv("sample_data.csv")

def data_preprocessing(model_checkpoint):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    print(tokenizer("Hello, this one sentence!", "And this sentence goes with it."))
    

if __name__ == "__main__":

    task = "mnli"
    model_checkpoint="distilbert-base-uncased"
    batch_size = 16

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)

    data_preprocessing(model_checkpoint)






