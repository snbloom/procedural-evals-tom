import os
import csv
import argparse
import random
from tqdm import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
STORIES_FILE = '../tinystories_words/tinystories_rows.txt'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/')
CSV_NAME = os.path.join(DATA_DIR, 'tinytom/')
FOLDER_NAMES = ["0_forward_action_false_belief", "0_forward_action_false_control", "0_forward_action_true_belief", "0_forward_action_true_control",
                "0_forward_belief_false_belief", "0_forward_belief_false_control", "0_forward_belief_true_belief", "0_forward_belief_true_control"]

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="three_words", help="generate conditions for which set of words/features")

def get_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0.5,
        max_tokens=450,
        n=1,
        request_timeout=180
    )
    return llm

def get_random_tinystory():
    with open(STORIES_FILE, "r") as f:
        stories = f.readlines()
        s = random.choice(stories).strip()
        return s

def get_formatted_instructions(tinytom_story):
    with open(PROMPT_DIR+"/tinytomstories.txt", "r") as f:
        instructions = f.read()
        instructions = instructions.replace("[tinytom_story]", tinytom_story)
        instructions = instructions.replace("[random_tinystory]", get_random_tinystory())
    return instructions

def convert_trimmed_stories(args):
    llm = get_llm()
    for folder_name in FOLDER_NAMES:
        folder_p = CONDITION_DIR + "/" + folder_name
        with open(folder_p + "/stories.csv", "r") as f:
            for template in f.readlines():
                template = template.split(";")
                story = template[0].strip()
                ending = " ".join(template[2].strip().split()[:2])
                instructions = get_formatted_instructions(story)
                system_message = SystemMessage(content=instructions)

                # print(instructions)
                # print("ending:", ending)

                messages = [system_message]
                responses = llm.generate([messages])

                for g, generation in enumerate(responses.generations[0]):
                    generated_story = generation.text.strip() 
                    generated_story = generated_story.replace("\n", " ")
                    generated_story = generated_story + " " + ending
                    print(generated_story)

                    # write to csv file
                    with open(folder_p + '/converted.txt', 'a') as f_w:
                        f_w.write(generated_story)
                        f_w.write("\n")
                        # writer = csv.writer(f_w)
                        # writer.writerow(generated_story)
                    
    



if __name__ == "__main__":  
    args = parser.parse_args()
    if args.method in ["three_words", "three_words_plus_features", "one_word", "no_forced_vocab"]:
        CONDITION_DIR += args.method
        CSV_NAME = CSV_NAME + "tinytom_" + args.method + ".csv"
    elif args.method == "bigtom":
        CONDITION_DIR += args.method
        CSV_NAME = os.path.join(DATA_DIR, 'bigtom/bigtom.csv')
    else: raise Exception("invalid method argument")

    convert_trimmed_stories(args)