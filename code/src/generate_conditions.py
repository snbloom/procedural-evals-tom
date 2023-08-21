import os
import csv
import argparse
import random

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/')
STORIES_FILE = '../tinystories_words/tinystories_rows.txt'
CSV_NAME = os.path.join(DATA_DIR, 'tinytom/')

INITIAL_BELIEF = [0, 1] # 0 hide initial belief, 1 show initial belief
VARIABLES = ['forward_belief', 'forward_action', 'backward_belief', 'percept_to_belief']
CONDITIONS = ['true_belief', 'false_belief', 'true_control', 'false_control']

FOLDER_NAMES = ["0_forward_belief_false_belief", "0_forward_belief_false_control", "0_forward_belief_true_belief", "0_forward_belief_true_control"]

MODEL = "gpt-4-0613"

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="tinytom", help="generate conditions for which set of words/features")

def get_llm():
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.0,
        max_tokens=450,
        n=1,
        request_timeout=180
    )
    return llm

def get_stories():
    with open(CSV_NAME, "r") as f:
        reader = csv.reader(f, delimiter=";")
        stories = list(reader)
    return stories

def get_formatted_instructions(tinytom_story, obj):
    with open(PROMPT_DIR+"/tinytomstories.txt", "r") as f:
        instructions = f.read()
        instructions = instructions.replace("[tinytom_story]", tinytom_story)
        instructions = instructions.replace('[object]', obj)
        print("instructions:", instructions)
    return instructions

def generate_conditions(stories):
    list_var = ["Story", "Aware of event", "Not Aware of event", "Action aware", "Action not aware", "Belief Question", "Desire Question", "Action Question",
                "Belief Answer Aware", "Desire Answer Aware", "Action Answer Aware", "Belief Answer not Aware",
                "Desire Answer not Aware", "Action Answer not Aware", "Random Event", "Aware of random event", "Not aware of random event", "Agent name", "Object"]

    for story_parts in stories:
        llm = get_llm()

        # # get number of existing stories
        # with open(folder_p + '/converted.txt', 'r') as f_r:
        #     if f_r.readable():
        #         print("readable")
        #         lines = f_r.readlines()
        #         print(lines)
        #         start_idx = len(lines)
        #     else: start_idx = 0
        # print("start_idx", start_idx)

        # just 0_ for now
        initial_belief = 0

        # just forward belief for now
        variable = 'forward_belief'

        nugget = story_parts[0]
        nugget_parts = nugget.split(".")
        observes_sentence = story_parts[1]
        does_not_observe_sentence = story_parts[2]
        obj = story_parts[-3]
        name = story_parts[-4]
        random_event = story_parts[-7]
        observes_random_sentence = story_parts[-6]
        does_not_observe_random_sentence = story_parts[-5]
        ending = name + " thinks that the " + obj + " is"
        print("ending", ending)

        belief_converted_nugget = ""
        control_converted_nugget = ""

        for condition in CONDITIONS:
            folder_name = str(initial_belief) + "_" + variable + "_" + condition

            # belief nugget for forward belief
            if "belief" in condition:
                to_convert = nugget

                # generate nugget conversion for BELIEF
                if belief_converted_nugget == "":
                    instructions = get_formatted_instructions(to_convert, obj)
                    print(instructions)
                    system_message = SystemMessage(content=instructions)
                    messages = [system_message]
                    responses = llm.generate([messages])

                    for g, generation in enumerate(responses.generations[0]):
                        generated_story = generation.text.strip() 
                        generated_story = generated_story.replace("\n", " ")
                        print(generated_story)
                        converted = generated_story
                        belief_converted_nugget = converted

                # add in ending
                if "true" in condition: converted = belief_converted_nugget + " " + observes_sentence + " " + ending
                else: converted = belief_converted_nugget + " " + does_not_observe_sentence + " " + ending
                
                        
            # control nugget for forward belief
            else:
                to_convert = ".".join(nugget_parts[:4]) + ". " + random_event

                # generate nugget conversion for CONTROL
                if control_converted_nugget == "":
                    instructions = get_formatted_instructions(to_convert, obj)
                    print(instructions)
                    system_message = SystemMessage(content=instructions)
                    messages = [system_message]
                    responses = llm.generate([messages])

                    for g, generation in enumerate(responses.generations[0]):
                        generated_story = generation.text.strip() 
                        generated_story = generated_story.replace("\n", " ")
                        print(generated_story)
                        converted = generated_story
                        control_converted_nugget = converted

                # add in ending
                if "true" in condition: converted = control_converted_nugget + " " + observes_random_sentence + " " + ending
                else: converted = control_converted_nugget + " " + does_not_observe_random_sentence + " " + ending
                

            print(converted)

            # write to file
            folder_p = CONDITION_DIR + "/" + folder_name
            print(folder_p)
            with open(folder_p + '/converted.txt', 'a') as f_w:
                f_w.write(converted)
                f_w.write("\n")
    


if __name__ == "__main__":  
    args = parser.parse_args()
    # Note: this only works for tinytom and bigtom... other methods are deprecated.
    if args.method == "tinytom":
        CONDITION_DIR += args.method
        CSV_NAME = CSV_NAME + "tinytom.csv"
    elif args.method in ["three_words", "three_words_plus_features", "one_word", "no_forced_vocab"]:
        CONDITION_DIR += args.method
        CSV_NAME = CSV_NAME + "tinytom_" + args.method + ".csv"
    elif args.method == "bigtom":
        CONDITION_DIR += args.method
        CSV_NAME = os.path.join(DATA_DIR, 'bigtom/bigtom.csv')
    else: raise Exception("invalid method argument")
    stories = get_stories()
    generate_conditions(stories)
