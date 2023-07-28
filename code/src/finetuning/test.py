import argparse

import torch
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/scr/kanishkg/models/finetuned-28/checkpoint-30")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipe = pipeline(
    "text-generation", model=args.model, device=device
)
txt = 'Once upon a time, Daisy went to a fun party with her friends. They were all very hungry and wanted to eat a yummy meal. But oh no! The meal was not ready yet, it was still cold. Daisy told her friends, "The meal is cold and not ready to eat." While they were talking and having fun, there was a sound. It was the oven timer! That meant the meal was hot and ready to eat. Daisy heard the oven timer. Daisy thinks that the meal is'
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
