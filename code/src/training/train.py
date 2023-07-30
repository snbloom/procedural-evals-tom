import os
import json
import random
import argparse

# import wandb
import torch
from transformers import LlamaModel, LlamaConfig
from transformers import LlamaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="../../configs/conf.json")

args = parser.parse_args()

# read config from a json config file
with open(args.config, "r") as f:
    config = json.load(f)

wandb.init(project="tiny-tom", dir='/scr/kanishkg/wandb/', name=config["name"], config=config)

# set seeds
random.seed(config["seed"])
torch.manual_seed(config["seed"])

# read json config file
if config["model"] == '250':
    with open("../../configs/llama-250.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == '115':
    with open("../../configs/llama-115.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == '43':
    with open("../../configs/llama-43.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == '14':
    with open("../../configs/llama-14.json", "r") as f:
        model_config = json.load(f)
else:
    raise ValueError("Invalid model size")

# pass the config to llama config and load llama model
model_config = LlamaConfig(**model_config) 
model = LlamaModel(model_config)
print(f"Number of parameters: {model.num_parameters()}")

# load tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


# load data
data_files = []
data_files += [f"{config['tinystories_dir']}/{f}" for f in os.listdir(config["tinystories_dir"]) if f.endswith(".json")]
# TODO: add tinytom-pretrain here if needed

# load data from jsons with hf datasets
hf_dataset = load_dataset("json", data_files=data_files)
# split into train, val, test
train_testval = hf_dataset.train_test_split(test_size=config["test_ratio"], seed=config["seed"])
test_val = train_testval["test"].train_test_split(test_size=0.5, seed=config["seed"])
hf_datasets = DatasetDict({
    "train": train_testval["train"],
    "val": test_val["test"],
    "test": test_val["train"]
})

context_length = config["context_length"]
def tokenize(element):
    element["story"] = element["story"].strip()
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


tokenized_datasets = hf_datasets.map(
    tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
)
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
    # check multi-gpu stage-2 if model fits, stage-3 if model doesn't fit
    # sharded_ddp=config["sharded_ddp"],
    # check if deepspeed is needed
    # deepspeed=config["deepspeed"]
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