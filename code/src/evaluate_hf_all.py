import argparse
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain import HuggingFaceHub


parser = argparse.ArgumentParser()

# model args
# parser.add_argument('--model', type=str, default='33', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=200, help='max tokens')

# eval args
# parser.add_argument('--num', '-n', type=int, default=1, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--local', action='store_true', default=True, help='local eval using transformers instead of huggingface hub')
parser.add_argument("--model_ids", type=list, default=["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M"], help="model ids")

# data args
parser.add_argument('--data_dir', type=str, default='../../data/conditions/three_words', help='data directory')
parser.add_argument('--variable', type=str, default='belief')
parser.add_argument('--condition', type=str, default='true_belief')
parser.add_argument('--init_belief', type=str, default="0_forward")

args = parser.parse_args()

variables = ["belief", "action"]
conditions = ["true_belief", "false_belief"]
init_beliefs = ["0_forward", "0_backward", "1_forward", "1_backward"]

model_ids = ["roneneldan/TinyStories-28M"]

LOG_FILE = f"../../data/evaluations.csv"

# Evaluate using both tinystories models
for repo_id in model_ids:
    if not args.local:
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":args.temperature, "max_length":args.max_tokens})
    else:
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)


    DATA_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/stories.csv"
    CONVERTED_FILE = f"{args.data_dir}/{args.init_belief}_{args.variable}_{args.condition}/converted.txt"

    correct_answers = []
    incorrect_answers = []
    inconsistent_unrelated_answers = []
    consistent_unrelated_answers = []
    partial_correct_answers = []

    count_correct = 0
    count_incorrect = 0
    count_partial = 0
    count_unrelated_consistent = 0
    count_unrelated_inconsistent = 0

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
            output = model.generate(input_ids, max_length=args.max_tokens, num_beams=1, )
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        # manual check for now
        # print(f"Story: {story}")
        # print(f"Question: {question}")
        # print(f"Correct Answer: {correct_answer}")
        # print(f"Wrong Answer: {wrong_answer}")
        print()
        # print(f"Converted Story: {repr(converted_story)}")
        print(f"Prediction {i}: {prediction}")
        print()
        while True:
            grade = input("Is the prediction correct? (y:yes/n:no/p:partial/c:unrelated-consistent/i:unrelated-inconsistent)")
            if grade == 'y' or grade=='yes':
                count_correct += 1
                correct_answers.append(prediction)
            elif grade == 'n' or grade=='no':
                count_incorrect += 1
                incorrect_answers.append(prediction)
            elif grade == 'p' or grade=='partial':
                count_partial += 1
                partial_correct_answers.append(prediction)
            elif grade == 'c' or grade=='unrelated-consistent':
                count_unrelated_consistent += 1
                consistent_unrelated_answers.append(prediction)
            elif grade == 'i' or grade=='unrelated-inconsistent':
                count_unrelated_inconsistent += 1
                inconsistent_unrelated_answers.append(prediction)
            else:
                continue
            break
        print(f"Current Tallies: correct {count_correct}, incorrect {count_incorrect}, partial {count_partial}, unrelated-consistent {count_unrelated_consistent}, unrelated_inconsistent {count_unrelated_inconsistent}")

    print(f"Final Tallies: correct {count_correct}, incorrect {count_incorrect}, partial {count_partial}, unrelated-consistent {count_unrelated_consistent}, unrelated_inconsistent {count_unrelated_inconsistent}")
    print("LOGGING OUTPUTS FOR MODEL", repo_id)
    with open(LOG_FILE, "a") as f_a:
        writer = csv.writer(f_a, delimiter=";")
        writer.writerow([repo_id, args.init_belief, args.variable, args.condition, count_correct, count_incorrect, count_partial, count_unrelated_consistent, count_unrelated_inconsistent, correct_answers, incorrect_answers, partial_correct_answers, consistent_unrelated_answers, count_unrelated_inconsistent, args])