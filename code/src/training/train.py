import os
import json
import random
import argparse

import wandb
import torch
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM, GPTNeoConfig, GPTNeoForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset

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
if config["model"] == 'l250':
    with open("../../configs/llama-250.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == 'l115':
    with open("../../configs/llama-115.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == 'l43':
    with open("../../configs/llama-43.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == 'l14':
    with open("../../configs/llama-14.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == 'n28':
    with open("../../configs/gpt-neo-28.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == 'n125':
    with open("../../configs/gpt-neo-125.json", "r") as f:
        model_config = json.load(f)
elif config["model"] == 'n250':
    with open("../../configs/gpt-neo-250.json", "r") as f:
        model_config = json.load(f)
else:
    raise ValueError("Invalid model name")

# pass the config to llama config and load llama model
if "l" in config["model"]:
    model_config = LlamaConfig(**model_config) 
    model = LlamaForCausalLM(model_config)
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    print(f"Number of parameters: {model.model.num_parameters()}")
elif "n" in config["model"]:
    model_config = GPTNeoConfig(**model_config)
    model = GPTNeoForCausalLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    print(f"Number of parameters: {model.num_parameters()-model_config.hidden_size*model_config.vocab_size}")



# load tokenizer


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

elif config["data"] == "no_think_believe":
    # load data from hf datasets
    train_file = os.path.join(config["tinystories_dir"], "train_no_think_believe.json")
    val_file = os.path.join(config["tinystories_dir"], "val_no_think_believe.json")

    hf_datasets = load_dataset(
            "json", 
            data_files={"train": train_file, "val": val_file}
                                                )

elif config["data"] == "v1":
    train_file = os.path.join(config["tinystories_dir"], "train.json")
    val_file = os.path.join(config["tinystories_dir"], "val.json")
    
    hf_datasets = load_dataset(
            "json", 
            data_files={"train": train_file, "val": val_file}
                                                )

context_length = config["context_length"]
def tokenize(element):
    if config["data"] == "full":
        stories = [tokenizer.bos_token + e.strip() for e in element["story"]]
    else:
        stories = [tokenizer.bos_token + e.strip() for e in element["text"]]
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
    # fp16=True,
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

# eval
if config["dataset"] == "full":
    trainer.predict(tokenized_datasets["test"])
