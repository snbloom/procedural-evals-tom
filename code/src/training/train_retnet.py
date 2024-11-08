import os
import random
import argparse
import json

import torch
from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorForLanguageModeling)
from transformers import LlamaTokenizerFast
from datasets import load_dataset, DatasetDict
import wandb

from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_json


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

if config["model"] == '43':
    model_config = load_config_from_json("../../configs/retnet-43.json")
else:
    raise ValueError("Invalid model size")
model = RetNetModelWithLMHead(model_config)
print(f"Number of parameters: {model.model.num_parameters()}")


# load data
if config["data"] == "full":
    data_files = []
    data_files += [f"{config['tinystories_dir']}/{f}" for f in os.listdir(config["tinystories_dir"]) if f.endswith(".json")]
    print(f"len data = {len(data_files)}")
    # TODO: add tinytom-pretrain here if needed

    # load data from jsons with hf datasets
    hf_dataset = load_dataset("json", data_files=data_files)
    print(hf_dataset)
    # split into train, val, test
    train_testval = hf_dataset["train"].train_test_split(test_size=config["test_ratio"], seed=config["seed"])
    test_val = train_testval["test"].train_test_split(test_size=0.5, seed=config["seed"])
    hf_datasets = DatasetDict({
        "train": train_testval["train"],
        "val": test_val["test"],
        "test": test_val["train"]
})
elif config["data"] == "gpt-4":
    # load data from hf datasets
    train_file = os.path.join(config["tinystories_dir"], "train_gpt4.json")
    val_file = os.path.join(config["tinystories_dir"], "val_gpt4.json")

    hf_datasets = load_dataset(
            "json", 
            data_files={"train": train_file, "val": val_file}
                                                )

# TODO: consider changing to GPT-Neo or LLAMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.eos_token
tokenizer.unk_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token

context_length = config["context_length"]

def tokenize(element):
    if config["data"] == "full":
        stories = [e.strip() for e in element["story"]]
    else:
        stories = [e.strip() for e in element["text"]]
 
    outputs = tokenizer(
        stories,
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

print(hf_datasets)
tokenized_datasets = hf_datasets.map(
    tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
)
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
    ddp_find_unused_parameters=False
    # check multi-gpu stage-2 if model fits, stage-3 if model doesn't fit
    # sharded_ddp=config["sharded_ddp"],
#     fsdp="full_shard"
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
    "valid": tokenized_datasets["val"],
    }
)

# train
trainer.train()
