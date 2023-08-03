import argparse

import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/scr/kanishkg/models/llama-training-14-2/checkpoint-90000")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

input_ids = tokenizer.encode("", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=500, do_sample=False)
prediction = tokenizer.decode(output[0], skip_special_tokens=True)
print(prediction)



