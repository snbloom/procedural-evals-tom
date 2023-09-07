import os
import csv
import argparse
from tqdm import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage
)

MODEL = "gpt-4-0613"
DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
STORIES_FILE = '../tinystories_words/tinystories_rows.txt'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/tinytom')
CONVERTED_PARTS_NAME = "tinytom/tinytom_converted_parts.txt"
CSV_NAME = os.path.join(DATA_DIR, 'tinytom/tinytom.csv')
FOLDER_NAMES = ["0_forward_belief_false_belief", "0_forward_belief_false_control", "0_forward_belief_true_belief", "0_forward_belief_true_control",
                "1_forward_belief_false_belief", "1_forward_belief_false_control", "1_forward_belief_true_belief", "1_forward_belief_true_control"]
BACKWARD_FOLDER_NAMES = ["0_backward_belief_false_belief", "0_backward_belief_false_control", "0_backward_belief_true_belief", "0_backward_belief_true_control",
                "1_backward_belief_false_belief", "1_backward_belief_false_control", "1_backward_belief_true_belief", "1_backward_belief_true_control"]
ALL_FOLDER_NAMES = FOLDER_NAMES + BACKWARD_FOLDER_NAMES
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="tinytom-v3", help="[tinytom, tinytom-v3]")
parser.add_argument('--num', type=int, default=None, help="max number of stories to convert")
parser.add_argument('--verbose', action='store_true', help="when enabled, print out unconverted and converted fragments")
parser.add_argument('--no_print', action='store_true', help="when enabled, don't print anything to the console except tqdm progress")
parser.add_argument('--re_stitch', action='store_true', help="when enabled, re-stitch all stories in the converted files")
parser.add_argument('--output', type=str, default="converted", help="run for converted or corrected versions [converted, corrected]")


def get_llm():
    return ChatOpenAI(
        model=MODEL,
        temperature=0.0,
        max_tokens=450,
        n=1,
        request_timeout=180
    )

def get_tinytom_stories():
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

def get_num_already_converted():
    with open(f'{DATA_DIR}/{CONVERTED_PARTS_NAME}', 'r') as f_r:
        start_idx = 0
        if f_r.readable():
            lines = f_r.readlines()
            start_idx = len(lines)
    return start_idx

def get_num_already_stitched(method, folder_name, output_name):
    if not os.path.isfile(f'{DATA_DIR}/conditions/{method}/{folder_name}/{output_name}.txt'): return 0
    with open(f'{DATA_DIR}/conditions/{method}/{folder_name}/{output_name}.txt', 'r') as f_r:
        start_idx = 0
        if f_r.readable():
            lines = f_r.readlines()
            start_idx = len(lines)
    return start_idx       

def convert_story_parts(stories, start_idx, args):
    llm = get_llm()
    
    count = 1
    for i, story in enumerate(tqdm(stories)):

        # limit by num argument
        if args.num is not None and count > args.num: break

        # only convert for new stories
        if i >= start_idx:

            # parse unconverted parts
            context_unconverted = ".".join(story[0].split(".")[:3]).strip() + "."
            causal_event_unconverted = story[0].split(".")[4].strip() + "."
            random_event_unconverted = story[14].strip()
            
            # print out parts to convert
            if args.verbose:
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

            # print out converted parts
            if not args.no_print:
                print("Converted Context: ", context)
                print("Converted Causal Event: ", causal_event)
                print("Converted Random Event: ", random_event)
                print()

            # record converted parts
            with open(f'{DATA_DIR}/{CONVERTED_PARTS_NAME}', "a") as f:
                f.write(";".join([context, causal_event, random_event]))
                f.write('\n')

            count += 1     

def re_stitch_stories(stories, end_idx, args, output_name):
    for folder_name in ALL_FOLDER_NAMES:
        with open(f'{DATA_DIR}/conditions/{args.method}/{folder_name}/{output_name}.txt', 'w') as f:
            f.write("") 
    stitch_stories(stories, end_idx, args, output_name)

def stitch_stories(stories, end_idx, args, output_name):

    start_idx = {}
    for folder_name in ALL_FOLDER_NAMES: 
        start_idx[folder_name] = get_num_already_stitched(args.method, folder_name, output_name)

    for i, story in enumerate(tqdm(stories)):
        if i == end_idx: break

        # parse unconverted parts
        init_belief_yes = story[0].split(".")[3] + "."
        percieve_causal_yes = story[1].strip()
        percieve_causal_no = story[2].strip()
        percieve_random_yes = story[15].strip()
        percieve_random_no = story[16].strip()
        action_percieve = story[3].strip()
        action_no_percieve = story[4].strip()
        name = story[17].strip()
        obj = story[18].strip().lower()
        correct_answer = story[11]

        # parse corrected ending
        if "is" in correct_answer: ending = correct_answer.split("is")[0] + "is"
        elif "are" in correct_answer: ending = correct_answer.split("are")[0] + "are"
        elif obj.lower() in correct_answer.lower(): ending = correct_answer.split(obj.lower())[0] + obj.lower()
        else: raise Exception("Cannot find 'is' or 'are' or obj in correct sentence.")
        if args.output == "converted": 
            ending.replace("believe", "think")
            assert("think" in ending)
        print(ending)

        # get filename for converted parts
        if args.method == "tinytom-v3": filename = f'{DATA_DIR}/tinytom/v3/tinytom_converted_parts.txt'
        elif args.method == "tinytom": filename = f"{DATA_DIR}/tinytom/tinytom_converted_parts.txt"
        else: raise Exception("Unexpected method. Expected [tinytom, tinytom-v3]")

        with open(filename, "r") as f:
            # read context, causal event, random event
            data = f.readlines()
            context, causal_event, random_event = data[i].strip().split(";")

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
            print("Converted Context: ", context)
            print("Converted Causal Event: ", causal_event)
            print("Converted Random Event: ", random_event)
            print()

        # stitch combinations by condition
        for folder_name in ALL_FOLDER_NAMES:
            if args.verbose: print("Condition:", folder_name)

            if i >= start_idx[folder_name]:
                stitched = context

                # initial belief
                if "1_" in folder_name:
                    stitched = " ".join([context, init_belief_yes])

                # control event
                if "control" in folder_name: stitched = " ".join([stitched, random_event])
                # causal event
                else: stitched = " ".join([stitched, causal_event])

                if "forward" in folder_name:
                    # true/false belief/control
                    if "true_belief" in folder_name: stitched = " ".join([stitched, percieve_causal_yes])
                    elif "true_control" in folder_name: stitched = " ".join([stitched, percieve_random_yes])
                    elif "false_belief" in folder_name: stitched = " ".join([stitched, percieve_causal_no])
                    elif "false_control" in folder_name: stitched = " ".join([stitched, percieve_random_no])
                    else: raise Exception("Expected: [true_belief, false_belief, true_control, false_control] in folder name.")
                
                elif "backward" in folder_name:
                    if "true_" in folder_name: stitched = " ".join([stitched, action_percieve])
                    elif "false_" in folder_name: stitched = " ".join([stitched, action_no_percieve])
                    else: raise Exception("Expected: [true_, false_] in folder name.")

                # Free response prompt
                stitched = " ".join([stitched, ending])

                if args.verbose: print(stitched, '\n')

                # write to file
                if not os.path.exists(f'{CONDITION_DIR}/{folder_name}'):
                    os.makedirs(f'{CONDITION_DIR}/{folder_name}')
                with open(f'{CONDITION_DIR}/{folder_name}/{output_name}.txt', 'a') as f_w:
                    f_w.write(stitched)
                    f_w.write("\n")   
    

if __name__ == "__main__":  

    args = parser.parse_args()

    # only works for v2 and v3... all others are deprecated
    if args.method == "tinytom-v3":
        CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/tinytom-v3')
        CSV_NAME = os.path.join(DATA_DIR, 'tinytom/v3/tinytom.csv')
        CONVERTED_PARTS_NAME = 'tinytom/v3/tinytom_converted_parts.txt'
    elif args.method != "tinytom": raise Exception("invalid method argument")
    
    stories = get_tinytom_stories()
    if not args.no_print: print("Length of stories:", len(stories))

    start_idx = get_num_already_converted()
    convert_story_parts(stories, start_idx, args)

    end_idx = get_num_already_converted()
    if args.re_stitch: re_stitch_stories(stories, end_idx, args, args.output)
    else: stitch_stories(stories, end_idx, args, args.output)
