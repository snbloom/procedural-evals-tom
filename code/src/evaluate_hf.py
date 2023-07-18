import argparse
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig



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
parser.add_argument('--local', action='store_true', help='local eval using transformers instead of huggingface hub')

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/three_words', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")

args = parser.parse_args()

if args.model == '33':
    repo_id = "roneneldan/TinyStories-33M"
elif args.model == '28':
    repo_id = "roneneldan/TinyStories-28M"

if not args.local:
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.0, "max_length":100})
else:
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")



DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

with open(CONVERTED_FILE, 'r') as f:
    converted = f.readlines()

score = 0
for i in range(args.num):
    story, question, correct_answer, wrong_answer, _ = data[i + args.offset]
    converted_story = converted[i + args.offset].strip()

    # hacky way to elicit answers
    # start with first two words of correct answer
    # prompt = correct_answer.split()[:2]
    # prompt = " ".join(prompt)
    # story = f"{story} {prompt}"
    
    # predict answer
    if not args.local:
        prediction = llm(converted_story)
    else:
        input_ids = tokenizer.encode(converted_story, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, num_beams=1)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    # manual check for now
    print(f"Story: {story}")
    print(f"Question: {question}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Wrong Answer: {wrong_answer}")
    print(f"Converted Story: {repr(converted_story)}")
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

print(f"Final Score: {score}/{args.num}")