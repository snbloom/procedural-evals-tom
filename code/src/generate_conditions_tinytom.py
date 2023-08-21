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

MODEL = "gpt-4-0613"
DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
STORIES_FILE = '../tinystories_words/tinystories_rows.txt'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/tinytom')
CSV_NAME = os.path.join(DATA_DIR, 'tinytom/tinytom.csv')
FOLDER_NAMES = ["0_forward_belief_false_belief", "0_forward_belief_false_control", "0_forward_belief_true_belief", "0_forward_belief_true_control",
                "1_forward_belief_false_belief", "1_forward_belief_false_control", "1_forward_belief_true_belief", "1_forward_belief_true_control"]

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="tinytom", help="generate conditions for which set of words/features")
parser.add_argument('--num', type=int, default=50, help="max number of stories to convert")
parser.add_argument('--verbose', action='store_true', help="when enabled, print out unconverted and converted fragments")

def get_llm():
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.0,
        max_tokens=450,
        n=1,
        request_timeout=180
    )
    return llm

def get_unconverted_stories():
    with open(CSV_NAME, "r") as f:
        reader = csv.reader(f, delimiter=";")
        return list(reader)

def get_formatted_instructions(kind, unconverted):
    if kind=="context": file = PROMPT_DIR+"/tinytom_stories_context.txt"
    elif kind=="event": file = PROMPT_DIR+"/tinytom_stories_event.txt"
    with open(file, "r") as f:
        instructions = f.read()
        instructions = instructions.replace("[tinytom story part]", unconverted)
    return instructions

def get_generation(llm, instructions):
    system_message = SystemMessage(content=instructions)
    messages = [system_message]
    responses = llm.generate([messages])

    for g, generation in enumerate(responses.generations[0]):
        generated_story = generation.text.strip() 
        generated_story = generated_story.replace("\n", " ")
        return generated_story

def convert_trimmed_stories(stories, args):
    llm = get_llm()

    # get number of already-converted stories
    with open(f'{DATA_DIR}/tinytom/tinytom_converted_parts.txt', 'r') as f_r:
        start_idx = 0
        if f_r.readable():
            lines = f_r.readlines()
            start_idx = len(lines)
    print(f"Already converted {start_idx} stories")
    print()
    count = 1

    for i, story in enumerate(stories):

        # limit by num argument
        if count > args.num: break

        # only convert for new stories
        if i >= start_idx:

            # parse unconverted parts
            context_unconverted = ".".join(story[0].split(".")[:3]).strip() + "."
            causal_event_unconverted = story[0].split(".")[4].strip() + "."
            random_event_unconverted = story[14].strip()
            init_belief_yes = story[0].split(".")[3] + "."
            percieve_causal_yes = story[1].strip()
            percieve_causal_no = story[2].strip()
            percieve_random_yes = story[15].strip()
            percieve_random_no = story[16].strip()
            name = story[17].strip()
            obj = story[18].strip()
            
            # print out parts to convert
            print("Context:", context_unconverted)
            print("Causal Event:", causal_event_unconverted)
            print("Random Event:", random_event_unconverted)
            print()

            # get converted context
            instr_context = get_formatted_instructions("context", context_unconverted)
            context = get_generation(llm, instr_context)

            # get converted event -- causal
            instr_causal = get_formatted_instructions("event", causal_event_unconverted)
            causal_event = get_generation(llm, instr_causal)

            # get converted event -- random
            instr_randpm = get_formatted_instructions("event", random_event_unconverted)
            random_event = get_generation(llm, instr_randpm)

            # print out all parts
            if args.verbose:
                print("Initial Belief Present:", init_belief_yes)
                print("Percieves Causal Event:", percieve_causal_yes)
                print("Does Not Percieve Causal Event:", percieve_causal_no)
                print("Percieves Random Event:", percieve_random_yes)
                print("Does Not Percieve Random Event:", percieve_random_no)
                print("Name:", name)
                print("Object:", obj)
                print()

            # print out converted parts
            print("Converted Context: ", context)
            print("Converted Causal Event: ", causal_event)
            print("Converted Random Event: ", random_event)
            print()

            # record converted parts
            with open(f'{DATA_DIR}/tinytom/tinytom_converted_parts.txt', "a") as f:
                f.write(";".join([context, causal_event, random_event]))
                f.write('\n')

            # stitch combinations by condition
            for folder_name in FOLDER_NAMES:
                if args.verbose: print("Condition:", folder_name)
                stitched = context

                # initial belief
                if "1_" in folder_name:
                    stitched = " ".join([context, init_belief_yes])

                # control event
                if "control" in folder_name: stitched = " ".join([stitched, random_event])
                # causal event
                else: stitched = " ".join([stitched, causal_event])

                # true/false belief/control
                if "true_belief" in folder_name: stitched = " ".join([stitched, percieve_causal_yes])
                elif "true_control" in folder_name: stitched = " ".join([stitched, percieve_random_yes])
                elif "false_belief" in folder_name: stitched = " ".join([stitched, percieve_causal_no])
                elif "false_control" in folder_name: stitched = " ".join([stitched, percieve_random_no])
                else: raise Exception("Expected: [true_belief, false_belief, true_control, false_control] in folder name.")

                # Free response prompt
                stitched = " ".join([stitched, name, "thinks that the", obj, "is"])

                if args.verbose: print(stitched, '\n')

                # write to file
                with open(f'{CONDITION_DIR}/{folder_name}/converted.txt', 'a') as f_w:
                    f_w.write(stitched)
                    f_w.write("\n")
            count += 1        
    

if __name__ == "__main__":  
    args = parser.parse_args()
    if args.method != "tinytom": raise Exception("invalid method argument")
    stories = get_unconverted_stories()
    print("Length of stories:", len(stories))
    convert_trimmed_stories(stories, args)