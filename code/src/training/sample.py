import argparse

import torch
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/scr/kanishkg/models/llama-training-14-2/checkpoint-90000")
args = parser.parse_args()

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

pipe = pipeline(
    "text-generation", model=args.model, device=device
)

print(pipe("", max_new_tokens=500, num_return_sequences=1)[0]["generated_text"])



