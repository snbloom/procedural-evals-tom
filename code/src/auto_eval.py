import argparse
import json
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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
# parser.add_argument('--num', '-n', type=int, default=1, help='number of evaluations')
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

variables = ["belief", "action"]
conditions = ["true_belief", "false_belief"]
init_beliefs = ["0_forward", "0_backward", "1_forward", "1_backward"]

all_model_ids = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M"]
model_id = args.model_id # or use the following shorthand:
if args.model_id == "33M": model_id = "roneneldan/TinyStories-33M"
if args.model_id == "28M": model_id = "roneneldan/TinyStories-28M"
if args.model_id == "gpt4": model_id = "gpt-4-0613"

LOG_FILE = "../../data/auto_evals.json"
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

def get_test_llm():
    test_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=args.max_tokens,
        n=1,
        request_timeout=180
    )
    return test_llm

eval_llm = get_eval_llm()

# get model (gpt4 vesus huggingface model)
if model_id =="gpt-4-0613":
    test_llm = get_test_llm()
else:
    if not args.local:
        test_llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":args.temperature, "max_new_tokens":args.max_tokens})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        # tokenizer = AutoTokenizer.from_pretrained(model_id)


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

with open(f"{PROMPT_DIR}/auto_eval_system.txt", "r") as f:
    sys_prompt = f.read()

print(sys_prompt)
print()

for i in range(len(converted)):
    story, question, correct_answer, wrong_answer, _ = data[i]
    converted_story = converted[i].strip()
    
    # predict answer
    if model_id =="gpt-4-0613":
        system_message = SystemMessage(content=converted_story)
        messages = [system_message]
        responses = test_llm.generate([messages])

        for g, generation in enumerate(responses.generations[0]):
            prediction = generation.text.strip() 
            prediction = prediction.replace("\n", " ")
            prediction = prediction.split(".")[0] + "."
    else:
        if not args.local:
            prediction = test_llm(converted_story)
            prediction = prediction[len(converted_story)+1:].split(".")[0] + "."
        else:
            input_ids = tokenizer.encode(converted_story, return_tensors="pt")
            output = model.generate(input_ids, max_new_tokens=args.max_tokens, num_beams=1, )
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = prediction[len(converted_story)+1:].split(".")[0] + "."

    # use gpt4 to check for accuracy
    with open(f"{PROMPT_DIR}/auto_eval_user.txt", "r") as f:
        user_prompt = f.read() 
        user_prompt = user_prompt.replace("[story]", converted_story)
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
            correct_answers.append(converted_story + " " + prediction)
        elif classification=="incorrect":
            count_incorrect += 1
            incorrect_answers.append(converted_story + " " + prediction)
        elif classification=="unrelated":
            count_unrelated += 1
            unrelated_answers.append(converted_story + " " + prediction)
        elif classification=="inconsistent":
            count_inconsistent += 1
            inconsistent_answers.append(converted_story + " " + prediction)
        else:
            raise Exception(f"Classification '{classification}' is not recognized")
        print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")


print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}, inconsistent {count_inconsistent}")
print("LOGGING OUTPUTS FOR MODEL", model_id)

with open(LOG_FILE, "r") as f:
    runs = json.load(f)

runs["evals"].append({
    "model_id":model_id,
    "method":"auto",
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