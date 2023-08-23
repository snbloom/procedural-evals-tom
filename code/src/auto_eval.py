import os
import subprocess
import argparse
import json
import csv
import torch
from crfm_llm import crfmLLM
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
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=20, help='max tokens')

# eval args
parser.add_argument('--num', '-n', type=int, default=50, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--local', action='store_true', default=True, help='local eval using transformers instead of huggingface hub')
parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-28M", help="gpt-4-0613, roneneldan/TinyStories-33M, roneneldan/TinyStories-28M")

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/tinytom', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")
parser.add_argument('--unconverted', action='store_true', help="whether to use unconverted (non tinystory-ified) versions")
parser.add_argument('--bigtom', action='store_true', help="run auto eval on bigtom dataset")

args = parser.parse_args()

all_model_ids = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M", 
                 "gpt-4-0613", "openai/text-davinci-003", "gpt-3.5-turbo",
                 "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500", "/scr/kanishkg/models/llama-training-43-4/checkpoint-50000",
                 '/scr/snbloom/models/finetuned-33-tinytom/checkpoint-125', '/scr/snbloom/models/finetuned-llama-43-tinytom/checkpoint-125']
open_ai_model_ids = ["gpt-4-0613", "openai/text-davinci-003", "gpt-3.5-turbo", "text-davinici-003"]

model_id = args.model_id # or use the following shorthand:
if args.model_id == "33M": model_id = "roneneldan/TinyStories-33M"
if args.model_id == "28M": model_id = "roneneldan/TinyStories-28M"
if args.model_id == "gpt4": model_id = "gpt-4-0613"
if args.model_id == "gpt35turbo": model_id = "gpt-3.5-turbo"
if args.model_id == "davinci003": model_id = "openai/text-davinci-003"

if args.model_id == "llama-14": model_id = "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500"
if args.model_id == "llama-43": model_id = "/scr/kanishkg/models/llama-training-43-4/checkpoint-50000"
if args.model_id == "finetuned-33": model_id = '/scr/snbloom/models/finetuned-33-tinytom/checkpoint-125'
if args.model_id == "finetuned-llama-43-100": model_id = '/scr/snbloom/models/finetuned-llama-43-tinytom-100/checkpoint-35'
if args.model_id == "finetuned-llama-43-400": model_id = '/scr/snbloom/models/finetuned-llama-43-tinytom-400/checkpoint-125'

data_range = f"{args.offset}-{args.offset + args.num}"

LOG_FILE = "../../data/evals.json"
PROMPT_DIR = "../prompt_instructions"

def get_llamac_prediction(prompt, args):
    # Store the original directory
    original_directory = os.getcwd()

    # Define the directory of the command
    command_directory = '/scr/kanishkg/llama2.c'

    # Change the current working directory
    os.chdir(command_directory)

    # Define the command and the arguments
    command = './run'
    arguments = [args.model_id, '-t', str(args.temperature), '-n', str(args.max_tokens), '-p', prompt]
    print(arguments)

    # Use subprocess.run() to run the command and capture the output
    completed_process = subprocess.run([command] + arguments, capture_output=True, text=True)
    # Check if the command ran successfully
    if completed_process.returncode != 0:
        print(f'The command failed with exit code {completed_process.returncode}')
        print(f'Standard error: {completed_process.stderr}')
    else:
        # Print or manipulate the command's output
        print(completed_process.stdout)
    
    # Change back to the original directory
    os.chdir(original_directory)
    return completed_process.stdout


def get_eval_llm():
    eval_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=250,
        n=1,
        request_timeout=180
    )
    return eval_llm

def get_test_llm(model):
    if model == "openai/text-davinci-003": return crfmLLM(model_name=model_id, temperature=args.temperature, max_tokens=args.max_tokens, verbose=False)
    else: return ChatOpenAI(
        model=model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        n=1,
        request_timeout=180
    )

eval_llm = get_eval_llm()

# get model (gpt4)
if model_id in open_ai_model_ids:
    test_llm = get_test_llm(model_id)
elif "bin" in model_id:
    test_llm = get_llamac_prediction
# get model finetuned / trained models or tinystories huggingface models)
else:
    if not args.local:
        test_llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":args.temperature, "max_new_tokens":args.max_tokens})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

data_dir = args.data_dir
RESULTS_DIR = os.path.join('../../data/results')
if args.bigtom: data_dir = '../../data/conditions/bigtom'
DATA_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
TRIMMED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories_trimmed.csv"
CONVERTED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

correct_answers = []
incorrect_answers = []
unrelated_answers = []
inconsistent_answers = []
predicted_answers = []
graded_answers = []

count_correct = 0
count_incorrect = 0
count_unrelated = 0
count_inconsistent = 0

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

if not args.bigtom: 
    with open(CONVERTED_FILE, 'r') as f:
        converted = f.readlines()

with open(TRIMMED_FILE, 'r') as f:
    trimmed = f.readlines()

with open(f"{PROMPT_DIR}/auto_eval_system.txt", "r") as f:
    sys_prompt = f.read()

print(sys_prompt)
print()

counter = 0
for i in range(len(data)):
    if i >= args.offset and counter < args.num:
        story, question, correct_answer, wrong_answer, _ = data[i]

        # story-ified story
        if not args.bigtom: converted_story = converted[i].strip()

        # non-story-ified version
        unconverted_story_parts = trimmed[i].split(';')
        unconverted_story = unconverted_story_parts[0] + " " + unconverted_story_parts[0].split()[0] + " thinks that the " + unconverted_story_parts[1].strip() + " is"
        
        # select converted or unconverted version (depending on args)
        if args.unconverted or args.bigtom: eval_story = unconverted_story
        else: eval_story = converted_story

        # predict answer
        if model_id in open_ai_model_ids:
            if model_id == "openai/text-davinci-003":
                response = test_llm(prompt=eval_story, stop=['.', '!', '\n'])
                prediction = response.split(".")[0] + "."
                prediction = prediction.replace("\n", " ")
            else:
                system_message = SystemMessage(content=eval_story)
                messages = [system_message]
                responses = test_llm.generate([messages])
                for g, generation in enumerate(responses.generations[0]):
                    prediction = generation.text.strip() 
                    prediction = prediction.split(".")[0] + "."
                    prediction = prediction.replace("\n", " ")
        elif "bin" in model_id:
            prediction = test_llm(eval_story, args)
            prediction = prediction[len(eval_story)+1:].split(".")[0] + "."
        else:
            if not args.local:
                prediction = test_llm(eval_story)
                prediction = prediction[len(eval_story)+1:].split(".")[0] + "."
                prediction = prediction.replace("\n", " ")
            else:
                input_ids = tokenizer.encode(eval_story, return_tensors="pt")
                output = model.generate(input_ids, max_new_tokens=args.max_tokens, num_beams=1, )
                prediction = tokenizer.decode(output[0], skip_special_tokens=True)
                prediction = prediction[len(eval_story)+1:].split(".")[0] + "."

        predicted_answers.append(prediction)
        # use gpt4 to check for accuracy
        with open(f"{PROMPT_DIR}/auto_eval_user.txt", "r") as f:
            user_prompt = f.read() 
            user_prompt = user_prompt.replace("[story]", eval_story)
            user_prompt = user_prompt.replace("[user_completion]", prediction)
            user_prompt = user_prompt.replace("[correct_completion]", correct_answer)
            user_prompt = user_prompt.replace("[incorrect_completion]", wrong_answer)

        print(user_prompt)

        system_message = SystemMessage(content=sys_prompt)
        user_msg = HumanMessage(content=user_prompt)
        messages = [system_message, user_msg]
        responses = eval_llm.generate([messages])

        for g, generation in enumerate(responses.generations[0]):
            eval = generation.text.strip() 
            print(eval)
            classification = eval.split("Evaluation:")[1].strip().lower()

            if classification=="correct":
                count_correct += 1
                correct_answers.append(eval_story + " " + prediction)
            elif classification=="incorrect":
                count_incorrect += 1
                incorrect_answers.append(eval_story + " " + prediction)
            elif classification=="unrelated":
                count_unrelated += 1
                unrelated_answers.append(eval_story + " " + prediction)
            elif classification=="inconsistent":
                count_inconsistent += 1
                inconsistent_answers.append(eval_story + " " + prediction)
            else:
                raise Exception(f"Classification '{classification}' is not recognized")
        graded_answers.append(classification)
        counter += 1
        print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")


print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")
print("LOGGING OUTPUTS FOR MODEL", model_id)

if args.bigtom: dataset = "bigtom"
else: dataset = "tinytom"

run = {
    "model_id":model_id,
    "method":"auto",
    "dataset": dataset,
    "unconverted": args.unconverted,
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
}

with open(LOG_FILE, "a") as f:
    json.dump(run, f)
    f.write('\n')

model_name = model_id.replace('/', '_')
model_id = model_id.replace('/', '_')
prediction = os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{args.condition}/auto_prediction_{model_id}_{args.temperature}_{args.variable}_{args.condition}_{args.offset}_{args.num}.csv')
accuracy_file = os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{args.condition}/auto_accuracy_{model_id}_{args.temperature}_{args.variable}_{args.condition}_{args.offset}_{args.num}.csv')

print("WRITING OUTPUTS TO", prediction, accuracy_file)

if not os.path.exists(os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{args.condition}')):
    os.makedirs(os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{args.condition}'))

with open(prediction, "w") as f:
    writer = csv.writer(f, delimiter=";")
    # write a new row per element in predicted answers 
    for predicted_answer in predicted_answers:
        writer.writerow([predicted_answer])

with open(accuracy_file, "w") as f:
    writer = csv.writer(f, delimiter=";")
    # write a new row per element in graded answers
    for graded_answer in graded_answers:
        writer.writerow([graded_answer])
