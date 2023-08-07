import argparse
import json
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--model', type=str, default='33', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=20, help='max tokens')

# eval args
parser.add_argument('--num', '-n', type=int, default=50, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--local', action='store_true', default=True, help='local eval using transformers instead of huggingface hub')
parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-28M", help="gpt-4-0613, roneneldan/TinyStories-33M, roneneldan/TinyStories-28M")

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/three_words', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")

args = parser.parse_args()

# all_model_ids = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M", 
#                  "gpt-4-0613", "text-davinci-003", "gpt-3.5-turbo",
#                  "/scr/kanishkg/models/finetuned-28-0r/checkpoint-45", "/scr/kanishkg/models/finetuned-33-0r/checkpoint-45",
#                  "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500", "/scr/kanishkg/models/llama-training-43-1/checkpoint-68500"]
# open_ai_model_ids = ["gpt-4-0613", "text-davinci-003", "gpt-3.5-turbo"]

# model_id = args.model_id # or use the following shorthand:
# if args.model_id == "33M": model_id = "roneneldan/TinyStories-33M"
# if args.model_id == "28M": model_id = "roneneldan/TinyStories-28M"
# if args.model_id == "gpt4": model_id = "gpt-4-0613"

data_range = f"{args.offset}-{args.offset + args.num-1}"

LOG_FILE = "../../data/fb_classifications.json"
PROMPT_DIR = "../prompt_instructions"

def get_eval_llm():
    eval_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=250,
        n=1,
        request_timeout=180
    )
    return eval_llm

# get model (gpt4)
eval_llm = get_eval_llm()


DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

correct_stories = []
incorrect_stories = []

correct_explanations = []
incorrect_explanations = []

count_correct = 0
count_incorrect = 0

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

with open(CONVERTED_FILE, 'r') as f:
    converted = f.readlines()

with open(f"{PROMPT_DIR}/false_belief_classifier.txt", "r") as f:
    sys_prompt = f.read()

print(sys_prompt)
print()

counter = 0
for i in range(len(converted)):
    if i >= args.offset and counter < args.num:
        story, question, correct_answer, wrong_answer, _ = data[i]
        converted_story = converted[i].strip()
        msg = "Story:\n" + converted_story
        print(msg)
        
        # get classification from gpt4
        system_message = SystemMessage(content=sys_prompt)
        user_message = HumanMessage(content=msg)
        messages = [system_message, user_message]
        responses = eval_llm.generate([messages])
        print(responses)
        for g, generation in enumerate(responses.generations[0]):
            prediction = generation.text.strip() 
            print(prediction)
            r_idx = prediction.index("Reasoning:")
            e_idx = prediction.index("Evaluation:")
            reasoning = prediction[r_idx:e_idx].strip()[10:]
            evaluation = prediction.split("Evaluation:")[1].strip().lower()
            print(reasoning)
            print(evaluation)
        
        # flag eval errors
        if args.condition not in ['true_belief', 'false_belief']: raise Exception("Unexpected evaluation. Expected ['false_belief', 'true_belief']")
        if evaluation not in ["false belief", "no false belief"]: raise Exception("Unexpected evaluation. Expected ['false belief', 'no false belief']") 

        # check for accuracy
        if args.condition == "false_belief":
            if evaluation == "no false belief": 
                count_incorrect += 1
                incorrect_stories.append(converted_story)
                incorrect_explanations.append(reasoning)
            else:
                count_correct += 1
                correct_stories.append(converted_story)
                correct_explanations.append(reasoning)
        else:
            if evaluation == "false belief": 
                count_incorrect += 1
                incorrect_stories.append(converted_story)
                incorrect_explanations.append(reasoning)
            else:
                count_correct += 1
                correct_stories.append(converted_story)
                correct_explanations.append(reasoning)


        print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}")
        counter += 1


print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}")
print("LOGGING OUTPUTS")

with open(LOG_FILE, "r") as f:
    runs = json.load(f)

runs["evals"].append({
    "dataset": "tinytom",
    "data_range":data_range,
    "init_belief":args.init_belief,
    "variable":args.variable,
    "condition":args.condition,
    "count_correct":count_correct,
    "count_incorrect":count_incorrect,
    "correct_stories":correct_stories,
    "incorrect_stories":incorrect_stories,
    "correct_explanations":correct_explanations,
    "incorrect_explanations":incorrect_explanations,
})
runs_json = json.dumps(runs)

if runs_json != "" and runs_json != "{}" and runs_json != "{'evals':[]}":
    with open(LOG_FILE, "w") as f:
        f.write(runs_json)