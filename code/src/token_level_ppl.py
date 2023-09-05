import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer



def get_token_ppl(sentence, bos=True):
    # Tokenize the sentence
    loss_fct = CrossEntropyLoss(reduction="none")

    input = tokenizer(
            sentence,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
    input_ids = input["input_ids"].to(device)
    attn_mask = input["attention_mask"].to(device)
    if bos:
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1).to(device)
        attn_mask = torch.cat(
                        [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                    )
    
    # Get log probabilities from the model
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask,  labels=input_ids).logits
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    # shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # ce_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
    perplexities = torch.exp(ce_loss)
    token_ids = shift_labels.cpu().numpy().tolist()[0]
    perplexities = perplexities.cpu().numpy().tolist()
    return token_ids, perplexities 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "roneneldan/TinyStories-28M"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    sentence = "Once upon a time"
    tokens, ppl  = get_token_ppl(sentence)
    for token, ppl in zip(tokens, ppl):
        print(tokenizer.decode(token), ppl)
