import argparse

import torch
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/scr/kanishkg/models/finetuned-28-0r/checkpoint-45")
args = parser.parse_args()

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

pipe = pipeline(
    "text-generation", model=args.model, device=device
)
txtfb = 'Once upon a time, Daisy went to a fun party with her friends. They were all very hungry and wanted to eat a yummy meal. But oh no! The meal was not ready yet, it was still cold. Daisy told her friends, "The meal is cold and not ready to eat." While they were talking and having fun, there was a sound. It was the oven timer! That meant the meal was hot and ready to eat. Daisy did not hear the oven timer as she was distracted by her friends. Daisy thinks that the meal is'
txttb = 'Once upon a time, Daisy went to a fun party with her friends. They were all very hungry and wanted to eat a yummy meal. But oh no! The meal was not ready yet, it was still cold. Daisy told her friends, "The meal is cold and not ready to eat." While they were talking and having fun, there was a sound. It was the oven timer! That meant the meal was hot and ready to eat. Daisy heard the oven timer. Daisy thinks that the meal is'
print(pipe(txttb, max_new_tokens=25, num_return_sequences=1)[0]["generated_text"])
print(pipe(txtfb, max_new_tokens=25, num_return_sequences=1)[0]["generated_text"])

