import json 
import os
import csv
import random
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from data_utils import get_tiny_tom, get_tiny_stories

# read args from a json config file
with open("config.json", "r") as f:
    args = json.load(f)

# set seeds
random.seed(args.seed)
torch.manual_seed(args.seed)

# load checkpoint for finetuning
if args.model == '33':
    repo_id = "roneneldan/TinyStories-33M"
elif args.model == '28':
    repo_id = "roneneldan/TinyStories-28M"

model = AutoModelForCausalLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# load data
# If the size of the new dataset is small,
# sample some of the existing dataset to create a new dataset
# this might help with overfitting
raw_datasets = {'train': [], 'val_tom': [], 'val_stories': []}

# load tinytom and preprocess
# NOTE: When the dataset is bigger, do this in a separate script,
# and load using hf datasets directly
tinytom = get_tiny_tom(args)
num_tiny_tom = sum([len(tinytom[cond]) for cond in args.conditions])

# load tinystories and preprocess
num_tiny_stories = num_tiny_tom * args.tinystories_ratio
tinystories = get_tiny_stories(args, num_tiny_stories)

# split tinytom into train and val
# 'train' (tinytom+tinystories)
for cond in args.conditions:
    num_train = int(len(tinytom[cond]) * args.train_ratio)
    num_val = len(tinytom[cond]) - num_train
    raw_datasets['train'] += [{"content": s} for s in tinytom[cond][:num_train]]
    raw_datasets['val_tom'] += [{"content": s} for s in tinytom[cond][num_train:]]

# split tinystories into train and val
raw_datasets['train'] += [{"content": s} for s in tinystories[:int(len(tinystories)*args.train_ratio)]]
raw_datasets['val_stories'] = [{"content": s} for s in tinystories[int(len(tinystories)*args.train_ratio):]]

hf_datasets = {split: Dataset.from_dict(data) for split, data in raw_datasets.items()}
del(raw_datasets)



# prepare datasets
context_length = args.context_length

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
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = hf_datasets.map(tokenize, batched=True)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# prepare training
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    logging_steps=args.log_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type=args.lr_scheduler_type,
    learning_rate=args.lr,
    save_steps=args.save_steps,
    fp16=True,
    push_to_hub=False,
    report_to="wandb",
    run_name=args.name,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

# train
trainer.train()
