import os
import csv
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

DATA_DIR = '../../data'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/')
CSV_NAME = os.path.join(DATA_DIR, 'tinytom/')
INITIAL_BELIEF = [0, 1] # 0 hide initial belief, 1 show initial belief
VARIABLES = ['forward_belief', 'forward_action', 'backward_belief', 'percept_to_belief']
CONDITIONS = ['true_belief', 'false_belief', 'true_control', 'false_control']

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="three_words", help="generate conditions for which set of words/features")

def get_eval_llm():
    return ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=250,
        n=1,
        request_timeout=180
    )

def get_completions():
    with open(CSV_NAME, "r") as f:
        reader = csv.reader(f, delimiter=";")
        completions = list(reader)
    return completions

def generate_conditions(completions):
    list_var = ["Story", "Aware of event", "Not Aware of event", "Action aware", "Action not aware", "Belief Question", "Desire Question", "Action Question",
                "Belief Answer Aware", "Desire Answer Aware", "Action Answer Aware", "Belief Answer not Aware",
                "Desire Answer not Aware", "Action Answer not Aware", "Random Event", "Aware of random event", "Not aware of random event", "Agent name", "Object"]

    llm = get_eval_llm()

    for completion_idx, completion in enumerate(completions):
        if completion_idx < 8: continue
        # get main story object for each story
        story = completion[0]
        print(story)
        name = story.split()[0]
        print(name)
        if args.method == "bigtom":
            system_message = SystemMessage(content="Your job is to identify the main object of a story provided by the user. Give your answer in 1-2 words. Do not include the word 'the'.")
            human_message = HumanMessage(content=story)
            messages = [system_message, human_message]
            responses = llm.generate([messages])
            for g, generation in enumerate(responses.generations[0]):
                obj = generation.text.strip().lower()

        print(obj)
        completion = completion[:-2]
        completion.append(name)
        completion.append(obj)

        print(completion)

        dict_var = {list_var[i]: completion[i] for i in range(len(list_var))}
        
        for init_belief in INITIAL_BELIEF:

            for variable in VARIABLES:  

                if variable == "percept_to_belief":
                    question = dict_var["Belief Question"]
                    answers = [dict_var["Belief Answer Aware"], dict_var["Belief Answer not Aware"]]
                    # story and parts 
                    story = dict_var["Story"]
                    story_parts = dict_var["Story"].split(".")
                    
                    if init_belief == 1:
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." 
                        

                elif variable == "forward_belief":
                    question = dict_var["Belief Question"]
                    answers = [dict_var["Belief Answer Aware"], dict_var["Belief Answer not Aware"]]
                    awareness = [dict_var["Aware of event"], dict_var["Not Aware of event"]]
                    awareness_random = [dict_var["Aware of random event"], dict_var["Not aware of random event"]]
                    
                    # story and parts 
                    story = dict_var["Story"]
                    story_parts = dict_var["Story"].split(".")

                    # include / exclude initial belief 
                    if init_belief == 0:
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." + story_parts[4] + "."
                        story_control = ".".join(story_parts[:3] + [" " + dict_var["Random Event"]])
                    elif init_belief == 1:
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." + story_parts[3] + "."  + story_parts[4] + "."
                        story_control = ".".join(story_parts[:4] + [" " + dict_var["Random Event"]])

                elif variable == "forward_action":
                    question = dict_var["Action Question"]
                    answers = [dict_var["Action Answer Aware"], dict_var["Action Answer not Aware"]]
                    awareness = [dict_var["Aware of event"], dict_var["Not Aware of event"]]
                    awareness_random = [dict_var["Aware of random event"], dict_var["Not aware of random event"]]
                    story = dict_var["Story"]
                    story_parts = dict_var["Story"].split(".")

                    # include / exclude initial belief 
                    if init_belief == 0:
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." + story_parts[4] + "."
                        story_control = ".".join(story_parts[:3] + [" " + dict_var["Random Event"]])
                    elif init_belief == 1:
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." + story_parts[3] + "."  + story_parts[4] + "."
                        story_control = ".".join(story_parts[:4] + [" " + dict_var["Random Event"]])

                elif variable == "backward_belief":
                    question = dict_var["Belief Question"]
                    answers = [dict_var["Belief Answer Aware"], dict_var["Belief Answer not Aware"]]
                    awareness = [dict_var["Aware of event"], dict_var["Not Aware of event"]]
                    actions = [dict_var["Action aware"], dict_var["Action not aware"]]
                    awareness_random = [dict_var["Aware of random event"], dict_var["Not aware of random event"]]

                    # include / exclude initial belief 
                    if init_belief == 0:
                        story_parts = dict_var["Story"].split(".")
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." + story_parts[4] + "."
                        story_parts = story.split(".")
                        story_control = ".".join(story_parts[:3] + [" " + dict_var["Random Event"]])

                    elif init_belief == 1:
                        
                        story_parts = dict_var["Story"].split(".")
                        story = story_parts[0] + "." + story_parts[1] + "." + story_parts[2] + "." +  story_parts[3] + "." + story_parts[4] + "."
                        story_parts = story.split(".")
                        story_control = ".".join(story_parts[:4] + [" " + dict_var["Random Event"]])

                elif variable == "backward_desire":
                    question = dict_var["Desire Question"]
                    answers = [dict_var["Desire Answer Aware"], dict_var["Desire Answer not Aware"]]
                    if answers[0] == answers[1]:
                        # if dict_var["Alternate Desire"] != "None": # @kankishk TODO: uncomment this line once we have added alternate desires for v_3
                            # answers[1] = dict_var["Alternate Desire"]
                        # else:
                        continue
                    awareness = [dict_var["Aware of event"], dict_var["Not Aware of event"]]
                    actions = [dict_var["Action aware"], dict_var["Action not aware"]]
                    awareness_random = [dict_var["Aware of random event"], dict_var["Not aware of random event"]]

                    # include / exclude initial belief 
                    if init_belief == 0:
                        story_parts = dict_var["Story"].split(".")
                        story = story_parts[0] + "." + story_parts[2] + "." + story_parts[4] + "."
                        story_parts = story.split(".")
                        story_control = ".".join(story_parts[:2] + [" " + dict_var["Random Event"]])

                    elif init_belief == 1:
                        
                        story_parts = dict_var["Story"].split(".")
                        story = story_parts[0] + "." + story_parts[2] + "." + story_parts[3] + "." + story_parts[4] + "."
                        story_parts = story.split(".")
                        story_control = ".".join(story_parts[:3] + [" " + dict_var["Random Event"]])

                for condition in CONDITIONS:

                    # check if folder exists

                    # Write the row to the CSV file based on the condition and variable
                    if variable == "percept_to_belief":
                        if condition == "true_belief" and init_belief == 1:
                            if not os.path.exists(os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}')):
                                os.makedirs(os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}'))
                            new_csv_file = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories.csv')
                            new_csv_file_trimmed = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories_trimmed.csv')
                            
                            with open(new_csv_file, "a" if completion_idx > 0 else "w", newline='') as csvfile:
                                writer = csv.writer(csvfile, delimiter=";")
                                writer.writerow([f"{story}", question, answers[0], answers[1], dict_var["Object"]])
                            with open(new_csv_file_trimmed, "a" if completion_idx > 0 else "w", newline='') as csvfile:
                                writer = csv.writer(csvfile, delimiter=";")
                                writer.writerow([f"{story}", question, answers[0], answers[1], dict_var["Object"]])
                          
                    elif variable != "percept_to_belief":
                        # Check if the new file needs to be created or appended
                        if not os.path.exists(os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}')):
                            os.makedirs(os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}'))
                        new_csv_file = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories.csv')
                        new_csv_file_trimmed = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories_trimmed.csv')
                        with open(new_csv_file, "a" if completion_idx > 0 else "w", newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=";")
                            if condition == "true_belief":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story} {awareness[0]} {actions[0]}", question, answers[0], answers[1], dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story} {actions[0]}", question, answers[0], answers[1], dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story} {awareness[0]}", question, answers[0], answers[1], dict_var["Object"]])
                            elif condition == "false_belief":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story} {awareness[1]} {actions[1]}", question, answers[1], answers[0], dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story} {actions[1]}", question, answers[1], answers[0], dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story} {awareness[1]}", question, answers[1], answers[0], dict_var["Object"]])
                            elif condition == "true_control":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story_control} {awareness_random[0]} {actions[1]}", question, answers[1], answers[0], dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story_control} {actions[1]}", question, answers[1], answers[0], dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story_control} {awareness_random[0]}", question, answers[1], answers[0], dict_var["Object"]])
                            elif condition == "false_control":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story_control} {awareness_random[1]} {actions[1]}", question, answers[1], answers[0], dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story_control} {actions[1]}", question, answers[1], answers[0], dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story_control} {awareness_random[1]}", question, answers[1], answers[0], dict_var["Object"]])
                        with open(new_csv_file_trimmed, "a" if completion_idx > 0 else "w", newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=";")
                            if condition == "true_belief":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story} {awareness[0]} {actions[0]}", dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story} {actions[0]}", dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story} {awareness[0]}", dict_var["Object"]])
                            elif condition == "false_belief":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story} {awareness[1]} {actions[1]}", dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story} {actions[1]}", dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story} {awareness[1]}", dict_var["Object"]])
                            elif condition == "true_control":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story_control} {awareness_random[0]} {actions[1]}", dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story_control} {actions[1]}", dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story_control} {awareness_random[0]}", dict_var["Object"]])
                            elif condition == "false_control":
                                if variable == "backward_desire":
                                    writer.writerow([f"{story_control} {awareness_random[1]} {actions[1]}", dict_var["Object"]])
                                elif variable == "backward_belief":
                                    writer.writerow([f"{story_control} {actions[1]}", dict_var["Object"]])
                                else:
                                    writer.writerow([f"{story_control} {awareness_random[1]}", dict_var["Object"]])


if __name__ == "__main__":  
    args = parser.parse_args()
    if args.method in ["three_words", "three_words_plus_features", "one_word", "no_forced_vocab"]:
        CONDITION_DIR += args.method
        CSV_NAME = CSV_NAME + "tinytom_" + args.method + ".csv"
    elif args.method == "bigtom":
        CONDITION_DIR += args.method
        CSV_NAME = os.path.join(DATA_DIR, 'bigtom/bigtom.csv')
    else: raise Exception("invalid method argument")
    completions = get_completions()
    generate_conditions(completions)
