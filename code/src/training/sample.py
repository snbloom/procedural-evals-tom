import argparse

import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/scr/kanishkg/models/llama-training-14-2/checkpoint-90000")
parser.add_argument("--config", type=str, default=None)

args = parser.parse_args()

if "ret" in args.model:
    from retnet.modeling_retnet import RetNetModelWithLMHead

    model = RetNetModelWithLMHead.from_pretrained(args.model)
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token


    inputs = tokenizer.encode("Once", return_tensors="pt")
 
    # parallel forward
    generated = model.generate(inputs, parallel_compute_prompt=True, bos_token_id=1, max_new_tokens=500)
    prediction = tokenizer.decode(generated)
    print(prediction)

else:
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    input_ids = tokenizer.encode("", return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=500, do_sample=False)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    print(prediction)



