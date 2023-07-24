import argparse
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig



from langchain import HuggingFaceHub


parser = argparse.ArgumentParser()

# model args
# parser.add_argument('--model', type=str, default='33', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=100, help='max tokens')

# eval args
# parser.add_argument('--num', '-n', type=int, default=1, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--local', action='store_true', default=True, help='local eval using transformers instead of huggingface hub')

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/three_words', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")

args = parser.parse_args()

variables = ["belief", "action"]
conditions = ["true_belief", "false_belief"]
init_beliefs = ["0_forward", "0_backward", "1_forward", "1_backward"]

model_ids = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M"]

LOG_FILE = f"../../data/evaluations.csv"

# Evaluate using both tinystories models
for repo_id in model_ids:
    if not args.local:
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.0, "max_length":120})
    else:
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)


    DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
    CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"
    correct_answers = []
    incorrect_answers = []
    maybe_answers = []

    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)

    with open(CONVERTED_FILE, 'r') as f:
        converted = f.readlines()

    score = 0
    for i in range(len(converted)):
        story, question, correct_answer, wrong_answer, _ = data[i]
        converted_story = converted[i].strip()
        
        # predict answer
        if not args.local:
            prediction = llm(converted_story)
        else:
            input_ids = tokenizer.encode(converted_story, return_tensors="pt")
            output = model.generate(input_ids, max_length=150, num_beams=1)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        # manual check for now
        # print(f"Story: {story}")
        # print(f"Question: {question}")
        # print(f"Correct Answer: {correct_answer}")
        # print(f"Wrong Answer: {wrong_answer}")
        print()
        # print(f"Converted Story: {repr(converted_story)}")
        print(f"Prediction: {prediction}")
        print()
        while True:
            grade = input("Is the prediction correct? (y:yes/n:no/m:maybe)")
            if grade == 'y' or grade=='yes':
                score += 1
                correct_answers.append(prediction)
            elif grade == 'm' or grade=="maybe":
                score += 0.5
                maybe_answers.append(prediction)
            elif grade == 'n' or grade=='no':
                score += 0
                incorrect_answers.append(prediction)
            else:
                continue
            break
        print(f"(Average) Running Score: {score}/{i+1} = {score/(i+1)}")

    print(f"Final (Average) Score: {score}/{len(converted)}")

    with open(LOG_FILE, "a") as f_a:
        writer = csv.writer(f_a, delimiter=";")
        writer.writerow([repo_id, args.init_belief, args.variable, args.condition, score/len(converted), correct_answers, incorrect_answers, maybe_answers])