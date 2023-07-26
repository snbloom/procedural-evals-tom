import json 
import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments



# read args from a json config file
with open("config.json", "r") as f:
    args = json.load(f)

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
# This might help with overfitting
# 'train' (tinytom+tinystories)
# 'val1' (tinytom)
# 'val2' (tinystories)
raw_datasets = {} # TODO: load stories from data file, and split into train, val, test


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

tokenized_datasets = raw_datasets.map(tokenize, batched=True)
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
