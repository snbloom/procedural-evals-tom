import argparse
import csv
import os

from langchain import HuggingFaceHub


parser = argparse.ArgumentParser()

# model args
parser.add_argument('--model', type=str, default='33', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=100, help='max tokens')

# eval args
parser.add_argument('--num', '-n', type=int, default=1, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/three_words', help='data directory')
parser.add_argument('--variable', type=str, default='action')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")

args = parser.parse_args()

if args.model == '33':
    repo_id = "roneneldan/TinyStories-33M"
elif args.model == '28':
    repo_id = "roneneldan/TinyStories-28M"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.0, "max_length":1})

DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

with open(CONVERTED_FILE, 'r') as f:
    converted = f.readlines()

score = 0
for i in range(args.num):
    _, question, correct_answer, wrong_answer = data[i + args.offset]
    story = converted[i + args.offset]

    # hacky way to elicit answers
    # start with first two words of correct answer
    # prompt = correct_answer.split()[:2]
    # prompt = " ".join(prompt)
    # story = f"{story} {prompt}"
    
    # predict answer
    prediction = llm(story, stop=[".", "?", "!", "\n"])

    # manual check for now
    print(f"Story: {story}")
    print(f"Question: {question}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Wrong Answer: {wrong_answer}")
    print(f"Prediction: {prediction}")
    while True:
        grade = input("Is the prediction correct? (y:yes/n:no/m:maybe)")
        if grade == 'y':
            score += 1
        elif grade == 'm':
            score += 0.5
        elif grade == 'n':
            score += 0
        else:
            continue
        break
    print(f"Score: {score}/{i+1} = {score/(i+1)}")

print(f"Final Score: {score}/{args.evaluations}")