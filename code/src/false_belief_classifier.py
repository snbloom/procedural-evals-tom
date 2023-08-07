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
STORIES_FILE = "../tinystories_words/tinystories_rows_gpt4.txt"

def get_eval_llm():
    return ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=250,
        n=1,
        request_timeout=180
    )

# get model (gpt4)
eval_llm = get_eval_llm()


DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

# correct_stories = []
# incorrect_stories = []

# correct_explanations = []
# incorrect_explanations = []

# count_correct = 0
# count_incorrect = 0

count_false_belief = 0
count_no_false_belief = 0

false_belief_stories = []
no_false_belief_stories = []

# with open(DATA_FILE, 'r') as f:
#     reader = csv.reader(f, delimiter=';')
#     data = list(reader)

# with open(CONVERTED_FILE, 'r') as f:
#     converted = f.readlines()

with open(STORIES_FILE, "r") as f:
    stories = f.readlines()

with open(f"{PROMPT_DIR}/false_belief_classifier.txt", "r") as f:
    sys_prompt = f.read()

print(sys_prompt)
print()

counter = 0
for i, story in enumerate(stories):
    if i >= args.offset and counter < args.num:
        # story, question, correct_answer, wrong_answer, _ = data[i]
        # converted_story = converted[i].strip()
        # converted_story = ".".join(converted_story.split(".")[:-1]) + "."
        # msg = "Story:\n" + converted_story
        # print(msg)
        
        msg = "Story:\n" + story
        print(msg)

        # get classification from gpt4
        system_message = SystemMessage(content=sys_prompt)
        user_message = HumanMessage(content=msg)
        messages = [system_message, user_message]
        responses = eval_llm.generate([messages])
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

        # log
        if evaluation == "false belief":
            count_false_belief += 1
            false_belief_stories.append({"story": story, "reasoning": reasoning})
        else:
            count_no_false_belief += 1
            no_false_belief_stories.append({"story": story, "reasoning": reasoning})
        # check for accuracy
        # if args.condition == "false_belief":
        #     if evaluation == "no false belief": 
        #         count_incorrect += 1
        #         incorrect_stories.append(converted_story)
        #         incorrect_explanations.append(reasoning)
        #     else:
        #         count_correct += 1
        #         correct_stories.append(converted_story)
        #         correct_explanations.append(reasoning)
        # else:
        #     if evaluation == "false belief": 
        #         count_incorrect += 1
        #         incorrect_stories.append(converted_story)
        #         incorrect_explanations.append(reasoning)
        #     else:
        #         count_correct += 1
        #         correct_stories.append(converted_story)
        #         correct_explanations.append(reasoning)

        print(f"Current Tallies: count_false_belief {count_false_belief}, count_no_false_belief {count_no_false_belief}")
        counter += 1


print(f"Final Tallies: count_false_belief {count_false_belief}, count_no_false_belief {count_no_false_belief}")
print("LOGGING OUTPUTS")

run = {
    "dataset": "tinystories-gpt4",
    "data_range":data_range,
    "count_false_belief":count_false_belief,
    "count_no_false_belief":count_no_false_belief,
    "false_belief_stories":false_belief_stories,
    "no_false_belief_stories":no_false_belief_stories
}

with open(LOG_FILE, "a") as f:
    json.dump(run, f)
    f.write("\n")