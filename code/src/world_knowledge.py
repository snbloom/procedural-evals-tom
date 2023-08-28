import os
import subprocess
import argparse
import json
import csv
import torch
from tqdm import tqdm
from crfm_llm import crfmLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel, PeftConfig

from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

INSTRUCTIONS = """Your job is to evaluate how much world knowledge is present in a given children's story. Classify the story as:

1 if it contains items or events that would be recognizable to a 1 year old.
2 if it contains items or events that would be recognizable to a 2 year old.
3 if it contains items or events that would be recognizable to a 3 year old.
...
20 if it contains items or events that would be recognizable to a 20 year old.

Give your answer in the following way:
Reasoning: <Reasoning>
Evaluation: <1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20>"""

def get_llm():
    return ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=450,
        n=1,
        request_timeout=180
    )

def get_generation(llm, instructions):
    system_message = SystemMessage(content=instructions)
    messages = [system_message]
    responses = llm.generate([messages])

    for g, generation in enumerate(responses.generations[0]):
        generated_story = generation.text.strip() 
        generated_story = generated_story.replace("\n", " ")
        return generated_story
    

