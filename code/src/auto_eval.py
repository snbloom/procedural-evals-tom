import os
import subprocess
import argparse
import json
import csv
from tqdm import tqdm
from crfm_llm import crfmLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=20, help='max tokens')
parser.add_argument('--beams', type=int, default=1, help='number of beams')
parser.add_argument('--lora', action='store_true')

# eval args
parser.add_argument('--num', '-n', type=int, default=50, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument("--no_print", action="store_true", help="print no intermediate steps")
parser.add_argument('--local', action='store_true', default=True, help='local eval using transformers instead of huggingface hub')
parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-28M", help="gpt-4-0613, roneneldan/TinyStories-33M, roneneldan/TinyStories-28M")

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/tinytom-v3', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
# parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")
parser.add_argument('--unconverted', action='store_true', help="whether to use unconverted (non tinystory-ified) versions")
parser.add_argument('--bigtom', action='store_true', help="run auto eval on bigtom dataset")
parser.add_argument('--filter', action='store_true', help="whether to filter out stories that are too long")
parser.add_argument('--think', action='store_true', help="whether to use think or believe")
parser.add_argument('--corrected', action='store_true', help="whether to use corrected stories")
parser.add_argument('--corrected_type', type=str, default="none", help="which filtered, corrected dataset type to use [in, out, none]")

args = parser.parse_args()

all_model_ids = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M", 
                 "gpt-4-0613", "openai/text-davinci-003", "gpt-3.5-turbo",
                 "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500", "/scr/kanishkg/models/llama-training-43-4/checkpoint-50000",
                 '/scr/snbloom/models/finetuned-33-tinytom/checkpoint-125', '/scr/snbloom/models/finetuned-llama-43-tinytom/checkpoint-125',
                 '/scr/snbloom/models/finetuned-28-tinytom-v2-100/checkpoint-80', '/scr/snbloom/models/finetuned-33-tinytom-v2-10']
open_ai_model_ids = ["gpt-4-0613", "openai/text-davinci-003", "gpt-3.5-turbo", "text-davinici-003"]

model_id = args.model_id # or use the following shorthand:
if args.model_id == "33": model_id = "roneneldan/TinyStories-33M"
if args.model_id == "28": model_id = "roneneldan/TinyStories-28M"
if args.model_id == "gpt4": model_id = "gpt-4-0613"
if args.model_id == "gpt35turbo": model_id = "gpt-3.5-turbo"
if args.model_id == "davinci003": model_id = "openai/text-davinci-003"

if args.model_id == "neo-125": model_id = "/scr/kanishkg/models/neo-training-125-1/checkpoint-28000"

if args.model_id == "llama-14": model_id = "/scr/kanishkg/models/llama-training-14-2/checkpoint-90500"
if args.model_id == "llama-43": model_id = "/scr/kanishkg/models/llama-training-43-2/checkpoint-91000"
if args.model_id == "finetuned-33": model_id = '/scr/snbloom/models/finetuned-33-tinytom/checkpoint-125'

if args.model_id == "finetuned-llama-43-100": model_id = '/scr/snbloom/models/finetuned-llama-43-tinytom-100/checkpoint-35'
if args.model_id == "finetuned-llama-43-200": model_id = '/scr/snbloom/models/finetuned-llama-43-tinytom-200/checkpoint-65'
if args.model_id == "finetuned-llama-43-400": model_id = '/scr/snbloom/models/finetuned-llama-43-tinytom-400/checkpoint-125'

if args.model_id == 'finetuned-28-v2-100': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v2-100/checkpoint-80'
if args.model_id == 'finetuned-28-v2-200': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v2-200/checkpoint-140'
if args.model_id == 'finetuned-28-v2-400': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v2-400/checkpoint-140'
if args.model_id == 'finetuned-28-v2-500': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v2-500/checkpoint-320'
if args.model_id == 'finetuned-28-v2-600': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v2-600/checkpoint-380'

if args.model_id == 'finetuned-33-v2-100': model_id = '/scr/snbloom/models/finetuned-33-tinytom-v2-100/checkpoint-140'
if args.model_id == 'finetuned-33-v2-200': model_id = "/scr/snbloom/models/finetuned-33-tinytom-v2-200/checkpoint-260"

if args.model_id == 'finetuned-28-v3-100': model_id = "/scr/snbloom/models/finetuned-28-tinytom-v3-100/checkpoint-80"
if args.model_id == 'finetuned-28-v3-200': model_id = "/scr/snbloom/models/finetuned-28-tinytom-v3-200/checkpoint-140"
if args.model_id == 'finetuned-28-v3-300': model_id = "/scr/snbloom/models/finetuned-28-tinytom-v3-300/checkpoint-180"
if args.model_id == 'finetuned-28-v3-400': model_id = "/scr/snbloom/models/finetuned-28-tinytom-v3-400/checkpoint-260"
if args.model_id == 'finetuned-28-v3-500': model_id = "/scr/snbloom/models/finetuned-28-tinytom-v3-500/checkpoint-260"
if args.model_id == 'finetuned-28-v3-600': model_id = "/scr/snbloom/models/finetuned-28-tinytom-v3-600/checkpoint-380"

if args.model_id == 'finetuned-28-v3-600-thinks': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v3-600-thinks/checkpoint-200'
if args.model_id == 'finetuned-28-v3-600-thinks-1_': model_id = '/scr/snbloom/models/finetuned-28-tinytom-v3-600-thinks-with-1_/checkpoint-380'

data_dir = args.data_dir
if data_dir == "v1": data_dir = "../../data/conditions/tinytom-v1"
if data_dir == "v3": data_dir = "../../data/conditions/tinytom-v3"

data_range = f"{args.offset+1}-{args.offset + args.num}"

LOG_FILE = "../../data/shorter_evals.json"
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
elif args.lora:
    config = PeftConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
else:
    if not args.local:
        test_llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":args.temperature, "max_new_tokens":args.max_tokens})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

RESULTS_DIR = os.path.join('../../data/results')
if args.bigtom: data_dir = '../../data/conditions/bigtom'

tb_answers, fb_answers = None, None
for condition in ["true_belief", "false_belief"]:
    DATA_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/stories.csv"
    TRIMMED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/stories_trimmed.csv"
    if args.corrected:
        CONVERTED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/corrected.txt"
    elif args.corrected_type == "in":
        CONVERTED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/corrected_in.txt"
        DATA_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/stories_in.csv"
    elif args.corrected_type == "out":
        CONVERTED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/corrected_out.txt"
        DATA_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/stories_out.csv"
    else:
        CONVERTED_FILE = f"{data_dir}/{args.init_belief}_{args.variable}_{condition}/converted.txt"
    if args.filter: FILTER_FILE = f"{data_dir}/ids_to_keep.txt"

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

    if args.filter:
        with open(FILTER_FILE, 'r') as f:
            ids_to_keep = f.readlines()
            ids_to_keep = [int(x.strip()) for x in ids_to_keep]
            # check if there are enough ids to keep
            if len(ids_to_keep) < args.num:
                raise Exception(f"Only {len(ids_to_keep)} ids to keep, but {args.num} evaluations requested")
            data = [data[i] for i in ids_to_keep]
            if not args.bigtom: converted = [converted[i] for i in ids_to_keep]
            trimmed = [trimmed[i] for i in ids_to_keep]

    with open(f"{PROMPT_DIR}/auto_eval_system.txt", "r") as f:
        sys_prompt = f.read()

    if args.verbose: 
        print(sys_prompt)
        print()

    counter = 0
    for i in tqdm(range(len(data))):
        if (i >= args.offset and counter < args.num) or args.filter:
            story, question, correct_answer, wrong_answer, _ = data[i]

            # story-ified story
            if not args.bigtom: converted_story = converted[i].strip()

            # non-story-ified version
            unconverted_story_parts = trimmed[i].split(';')
            unconverted_story = unconverted_story_parts[0] + " " + unconverted_story_parts[0].split()[0] + " believes that the " + unconverted_story_parts[1].strip() + " is"
            
            # select converted or unconverted version (depending on args)
            if args.unconverted or args.bigtom: eval_story = unconverted_story
            else: eval_story = converted_story
            
            if args.think:
                eval_story = eval_story.replace("believed", "thought")
                eval_story = eval_story.replace("believe", "think")
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
                    tokenizer.pad_token = tokenizer.bos_token
                    output = model.generate(input_ids=input_ids, max_new_tokens=args.max_tokens, num_beams=args.beams, )
                    prediction = tokenizer.decode(output[0], skip_special_tokens=False)
                    prediction = prediction[len(eval_story)+1:].split(".")[0] + "."


            predicted_answers.append(prediction)
            # use gpt4 to check for accuracy
            with open(f"{PROMPT_DIR}/auto_eval_user.txt", "r") as f:
                user_prompt = f.read() 
                user_prompt = user_prompt.replace("[story]", eval_story)
                user_prompt = user_prompt.replace("[user_completion]", prediction)
                user_prompt = user_prompt.replace("[correct_completion]", correct_answer)
                user_prompt = user_prompt.replace("[incorrect_completion]", wrong_answer)

            if not args.no_print: print(user_prompt)

            system_message = SystemMessage(content=sys_prompt)
            user_msg = HumanMessage(content=user_prompt)
            messages = [system_message, user_msg]
            responses = eval_llm.generate([messages])

            for g, generation in enumerate(responses.generations[0]):
                eval = generation.text.strip() 
                if not args.no_print: print(eval)
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
            if not args.no_print: print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")


    print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")
    print("LOGGING OUTPUTS FOR MODEL", model_id)

    if args.bigtom: dataset = "bigtom"
    elif data_dir == "../../data/conditions/tinytom-v1": dataset = "tinytom-v1"
    elif data_dir == "../../data/conditions/tinytom-v3": dataset = "tinytom-v3"
    else: dataset = "tinytom"

    run = {
        "model_id":model_id,
        "method":"auto",
        "dataset": dataset,
        "unconverted": args.unconverted,
        "data_range":data_range,
        "init_belief":args.init_belief,
        "corrected": args.corrected,
        "corrected_type": args.corrected_type,
        "variable":args.variable,
        "condition":condition,
        "count_correct":count_correct,
        "count_incorrect":count_incorrect,
        "count_unrelated":count_unrelated,
        "count_inconsistent":count_inconsistent,
        "incorrect_stories": incorrect_answers,
        "unrelated_stories": unrelated_answers,
        "inconsistent_stories": inconsistent_answers
    }

    with open(LOG_FILE, "a") as f:
        json.dump(run, f, indent=6)
        f.write('\n')

    model_name = model_id.replace('/', '_')
    model_id = model_id.replace('/', '_')
    model_id += f"_{args.beams}"

    if args.corrected or args.corrected_type == "in" or args.corrected_type == "out": co = "corrected"
    elif args.unconverted: co = "unconverted"
    else: co = "converted"

    if args.think: th = "think"
    else: th = "belief"
    prediction = os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{condition}_{co}_{args.corrected_type}/auto_prediction_{model_id}_{args.temperature}_{args.variable}_{condition}_{args.offset}_{args.num}_{th}.csv')
    accuracy_file = os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{condition}_{co}_{args.corrected_type}/auto_accuracy_{model_id}_{args.temperature}_{args.variable}_{condition}_{args.offset}_{args.num}_{th}.csv')

    print("WRITING OUTPUTS TO", prediction, accuracy_file)
    print(args.model_id, condition, args.init_belief, co)

    if not os.path.exists(os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{condition}_{co}_{args.corrected_type}')):
        os.makedirs(os.path.join(RESULTS_DIR, dataset, f'{args.init_belief}_{args.variable}_{condition}_{co}_{args.corrected_type}'))

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

    if "true" in condition:
        tb_answers = graded_answers
    else:
        fb_answers = graded_answers
    
print(tb_answers.count('correct'))
print(tb_answers.count('incorrect'))
print(tb_answers.count('unrelated'))
print(tb_answers.count('inconsistent'))
print(fb_answers.count('correct'))
print(fb_answers.count('incorrect'))
print(fb_answers.count('unrelated'))
print(fb_answers.count('inconsistent'))

for i in range(len(tb_answers)):
    if tb_answers[i] == 'correct':
        tb_answers[i] = True
    else:
        tb_answers[i] = False
    if fb_answers[i] == 'correct':
        fb_answers[i] = True
    else:
        fb_answers[i] = False



print(f'True Accuracy: {np.mean(tb_answers)}')
print(f'False Accuracy: {np.mean(fb_answers)}')
# take intersection of true and false accuracy
print(f'Intersection Accuracy: {np.mean(np.logical_and(tb_answers, fb_answers))}')