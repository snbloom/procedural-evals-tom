import json
import random
import torch
import argparse


import pandas as pd
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

from data_utils import get_tiny_tom, get_tiny_stories


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="../../configs/conf.json")

args = parser.parse_args()

# read config from a json config file
with open(args.config, "r") as f:
    config = json.load(f)

wandb.init(project="tiny-tom", dir='/scr/kanishkg/wandb/', config=config)

# set seeds
random.seed(config["seed"])
torch.manual_seed(config["seed"])

# load checkpoint for finetuning
if config["model"] == '33':
    repo_id = "roneneldan/TinyStories-33M"
elif config["model"] == '28':
    repo_id = "roneneldan/TinyStories-28M"

print(f"Loading model from {repo_id}")
model = AutoModelForCausalLM.from_pretrained(repo_id)
print("Model loaded")
print(f"Number of parameters: {model.num_parameters()}")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# load data
# If the size of the new dataset is small,
# sample some of the existing dataset to create a new dataset
# this might help with overfitting
print("Loading data")
raw_datasets = {'train': [], 'val_tom': [], 'val_stories': []}

# load tinytom and preprocess
# NOTE: When the dataset is bigger, do this in a separate script,
# and load using hf datasets directly
print("Loading tinytom")
tinytom = get_tiny_tom(config)
num_tiny_tom = sum([len(tinytom[cond]) for cond in config["conditions"]])
print(f"Number of tinytom stories: {num_tiny_tom}")

# load tinystories and preprocess
print("Loading tinystories")
num_tiny_stories = int(num_tiny_tom * config["tinystories_ratio"])
tinystories = get_tiny_stories(config, num_tiny_stories)

# split tinytom into train and val
# 'train' (tinytom+tinystories)
print("Splitting into train and val")
for cond in config["conditions"]:
    num_train = int(len(tinytom[cond]) * config["train_ratio"])
    num_val = len(tinytom[cond]) - num_train
    raw_datasets['train'] += [{"content": s} for s in tinytom[cond][:num_train]]
    raw_datasets['val_tom'] += [{"content": s} for s in tinytom[cond][num_train:]]

# split tinystories into train and val
raw_datasets['train'] += [{"content": s} for s in tinystories[:int(len(tinystories)*config["train_ratio"])]]
raw_datasets['val_stories'] = [{"content": s} for s in tinystories[int(len(tinystories)*config["train_ratio"]):]]

hf_datasets = DatasetDict({split: Dataset.from_pandas(pd.DataFrame(data=data)) for split, data in raw_datasets.items()})

print(hf_datasets)

# prepare datasets
print("Tokenizing datasets")
context_length = config["context_length"]

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = hf_datasets.map(tokenize, batched=True, remove_columns=hf_datasets["train"].column_names)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print("tokenized dataset", tokenized_datasets)


# prepare training
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    evaluation_strategy="steps",
    eval_steps=config["eval_steps"],
    logging_steps=config["log_steps"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=config["weight_decay"],
    warmup_steps=config["warmup_steps"],
    lr_scheduler_type=config["lr_scheduler_type"],
    learning_rate=config["lr"],
    save_strategy="steps",
    save_total_limit=config["save_total_limit"],
    save_steps=config["save_steps"],
    seed=config["seed"],
    fp16=True,
    push_to_hub=False,
    report_to="wandb",
    run_name=config["name"],
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset={
    "valid_tom": tokenized_datasets["val_tom"],
    "valid_stories": tokenized_datasets["val_stories"],
    }
)

# train
trainer.train()
