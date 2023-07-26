import argparse
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
parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-28M", help="gpt-4, roneneldan/TinyStories-33M, roneneldan/TinyStories-28M")

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
model_id = args.model_id

LOG_FILE = f"../../data/evaluations.csv"

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
# inconsistent_unrelated_answers = []
# consistent_unrelated_answers = []
# partial_correct_answers = []
unrelated_answers = []
inconsistent_answers = []

count_correct = 0
count_incorrect = 0
# count_partial = 0
# count_unrelated_consistent = 0
# count_unrelated_inconsistent = 0
count_unrelated = 0
count_inconsistent = 0

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

with open(CONVERTED_FILE, 'r') as f:
    converted = f.readlines()

for i in range(len(converted)):
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
    else:
        if not args.local:
            prediction = llm(converted_story)
            prediction = prediction[len(converted_story)+1:].split(".")[0] + "."
        else:
            input_ids = tokenizer.encode(converted_story, return_tensors="pt")
            output = model.generate(input_ids, max_new_tokens=args.max_tokens, num_beams=1, )
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = prediction[len(converted_story)+1:].split(".")[0] + "."

    # manual check for now
    # print(f"Story: {story}")
    # print(f"Question: {question}")
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
    print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}")

print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}, unrelated {count_unrelated}")
print("LOGGING OUTPUTS FOR MODEL", model_id)
with open(LOG_FILE, "a") as f_a:
    writer = csv.writer(f_a, delimiter=";")
    writer.writerow([model_id, args.init_belief, args.variable, args.condition, count_correct, count_incorrect, count_unrelated, correct_answers, incorrect_answers, unrelated_answers, args])