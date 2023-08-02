import argparse
import csv
import json
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
# parser.add_argument('--model', type=str, default='33', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=20, help='max tokens')

# eval args
parser.add_argument('--num', '-n', type=int, default=50, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--local', action='store_true', help='local eval using transformers instead of huggingface hub')
parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-28M", help="gpt-4-0613, roneneldan/TinyStories-33M, roneneldan/TinyStories-28M")

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/three_words', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")

args = parser.parse_args()

variables = ["belief", "action"]
conditions = ["true_belief", "false_belief"]
init_beliefs = ["0_forward", "0_backward", "1_forward", "1_backward"]

all_model_ids = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M", "gpt-4-0613", 
                 "/scr/kanishkg/models/finetuned-28-0r/checkpoint-45", "/scr/kanishkg/models/finetuned-33-0r/checkpoint-45",
                 "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500"]
our_models_ids = ["/scr/kanishkg/models/finetuned-28-0r/checkpoint-45", "/scr/kanishkg/models/finetuned-33-0r/checkpoint-45",
                 "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500"]
model_id = args.model_id # or use the following shorthand:
if args.model_id == "33M": model_id = "roneneldan/TinyStories-33M"
if args.model_id == "28M": model_id = "roneneldan/TinyStories-28M"
if args.model_id == "gpt4" or args.model_id == "gpt-4": model_id = "gpt-4-0613"
if args.model_id == "finetuned_28": model_id = "/scr/kanishkg/models/finetuned-28-0r/checkpoint-45"
if args.model_id == "finetuned_33": model_id = "/scr/kanishkg/models/finetuned-33-0r/checkpoint-45"
if args.model_id == "llama": model_id = "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500"

LOG_FILE = f"../../data/evals.json"

data_range = f"{args.offset}-{args.offset + args.num}"

def get_llm():
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.0,
        max_tokens=args.max_tokens,
        n=1,
        request_timeout=180
    )
    return llm

# get model (gpt4 vesus huggingface model)
if model_id =="gpt-4":
    llm = get_llm()
elif model_id in our_models_ids:
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline( "text-generation", model=model_id, device=device )
else:
    if not args.local:
        llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":args.temperature, "max_new_tokens":args.max_tokens})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer = AutoTokenizer.from_pretrained(model_id)


DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

correct_answers = []
incorrect_answers = []
unrelated_answers = []
inconsistent_answers = []

count_correct = 0
count_incorrect = 0
count_unrelated = 0
count_inconsistent = 0

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

with open(CONVERTED_FILE, 'r') as f:
    converted = f.readlines()

counter = 0
for i in range(len(converted)):
    if i >= args.offset and counter < args.num:
        print(i)
        story, question, correct_answer, wrong_answer, _ = data[i]
        converted_story = converted[i].strip()
        
        # predict answer
        if model_id =="gpt-4":
            system_message = SystemMessage(content=converted_story)
            messages = [system_message]
            responses = llm.generate([messages])

            for g, generation in enumerate(responses.generations[0]):
                prediction = generation.text.strip() 
                prediction = prediction.replace("\n", " ")
                prediction = prediction.split(".")[0] + "."
        elif model_id in our_models_ids:
            prediction = pipe(converted_story, num_return_sequences=1, max_new_tokens=20)[0]["generated_text"]
            prediction = prediction[len(converted_story)+1:].split(".")[0] + "."
            prediction = prediction.replace("\n", " ")
        else:
            if not args.local:
                prediction = llm(converted_story)
                prediction = prediction[len(converted_story)+1:].split(".")[0] + "."
            else:
                input_ids = tokenizer.encode(converted_story, return_tensors="pt")
                output = model.generate(input_ids, max_new_tokens=args.max_tokens, num_beams=1, )
                prediction = tokenizer.decode(output[0], skip_special_tokens=True)
                prediction = prediction[len(converted_story)+1:].split(".")[0] + "."

        # manual check 
        print()
        print(f"Story {i}: {repr(converted_story)}")
        print(f"Prediction: {prediction}")
        print()
        print(f"Correct Answer: {correct_answer}")
        print(f"Wrong Answer: {wrong_answer}")
        print()

        while True:
            grade = input("Is the prediction correct? (y:yes/n:no/u:unrelated/i:inconsistent) ")
            if grade == 'y' or grade=='yes':
                count_correct += 1
                correct_answers.append(converted_story + " " + prediction)
            elif grade == 'n' or grade=='no':
                count_incorrect += 1
                incorrect_answers.append(converted_story + " " + prediction)
            elif grade == 'u' or grade=='unrelated':
                count_unrelated += 1
                unrelated_answers.append(converted_story + " " + prediction)
            elif grade == 'i' or grade=='inconsistent':
                count_inconsistent += 1
                inconsistent_answers.append(converted_story + " " + prediction)
            else:
                continue
            break
        counter += 1
        print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")

print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")
print("LOGGING OUTPUTS FOR MODEL", model_id)

with open(LOG_FILE, "r") as f:
    runs = json.load(f)

runs["evals"].append({
    "model_id":model_id,
    "method":"manual",
    "data_range":data_range,
    "init_belief":args.init_belief,
    "variable":args.variable,
    "condition":args.condition,
    "count_correct":count_correct,
    "count_incorrect":count_incorrect,
    "count_unrelated":count_unrelated,
    "count_inconsistent":count_inconsistent,
    "correct_stories":correct_answers,
    "incorrect_stories":incorrect_answers,
    "unrelated_stories":unrelated_answers,
    "inconsistent_stories":inconsistent_answers,
})
runs_json = json.dumps(runs)
print(runs_json)

if runs_json != "" and runs_json != "{}" and runs_json != "{'evals':[]}":
    with open(LOG_FILE, "w") as f:
        f.write(runs_json)