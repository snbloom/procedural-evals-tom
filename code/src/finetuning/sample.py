import argparse

import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="/scr/kanishkg/models/finetuned-28-0r/checkpoint-45")
args = parser.parse_args()


txtfb = 'Once upon a time, Daisy went to a fun party with her friends. They were all very hungry and wanted to eat a yummy meal. But oh no! The meal was not ready yet, it was still cold. Daisy told her friends, "The meal is cold and not ready to eat." While they were talking and having fun, there was a sound. It was the oven timer! That meant the meal was hot and ready to eat. Daisy did not hear the oven timer as she was distracted by her friends. Daisy thinks that the meal is'
txttb = 'Once upon a time, Daisy went to a fun party with her friends. They were all very hungry and wanted to eat a yummy meal. But oh no! The meal was not ready yet, it was still cold. Daisy told her friends, "The meal is cold and not ready to eat." While they were talking and having fun, there was a sound. It was the oven timer! That meant the meal was hot and ready to eat. Daisy heard the oven timer. Daisy thinks that the meal is'
txtfb = "Uma and her friends were at the park, having lots of fun. Uma really wanted to go on the big slide. The slide was safe for kids. But then, something unexpected happened. A screw on the slide became loose. This made the slide dangerous. Uma does not see the screw come loose. Uma thinks that the slide is"
txttb = "Uma and her friends were at the park, having lots of fun. Uma really wanted to go on the big slide. The slide was safe for kids. But then, something unexpected happened. A screw on the slide became loose. This made the slide dangerous. Uma sees the loose screw become loose. Uma thinks that the slide is"
txtfb = "Once upon a time, there was a girl named Mae. Mae loved her garden very much. One day, she saw a worm in her garden. The worm was very juicy. But oh no! A hungry mole came and ate the worm. Now, there was no worm left for the birds to eat. Mae didn't see the mole eat the worm. Mae thinks that the worm is very tasty."
txtfb= "Pia is at her school library. She wants to read a mystery book. Pia has seen that the mystery book is up high on a shelf. But then a tall boy moves the mystery book to the bottom shelf. Pia doesn't see the tall boy move the book. Pia thinks that the mystery book is"
txttb= "Pia is at her school library. She wants to read a mystery book. Pia has seen that the mystery book is up high on a shelf. But then a tall boy moves the mystery book to the bottom shelf. Pia sees the tall boy move the book. Pia thinks that the mystery book is"

model = AutoModelForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

input_ids = tokenizer.encode(txtfb, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=500, do_sample=False, num_beams=5, pad_token_id=tokenizer.eos_token_id)
prediction = tokenizer.decode(output[0], skip_special_tokens=True)
print(prediction)

input_ids = tokenizer.encode(txttb, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=500, do_sample=False, num_beams=5, pad_token_id=tokenizer.eos_token_id)
prediction = tokenizer.decode(output[0], skip_special_tokens=True)
print(prediction)



# print(pipe("", max_new_tokens=100, num_return_sequences=1)[0]["generated_text"])

