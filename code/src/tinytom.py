import random
import csv
import tqdm
import argparse
import ast

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from utils import push_data, get_num_items, get_vars_from_out

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']
DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
WORDS_DIR = '../tinystories_words'
REPO_URL = 'https://github.com/cicl-stanford/marple_text'
CSV_NAME = 'tinytom/tinytom'
OBJECT_STATES_CSV = 'tinytom/object_states.csv'
DISCARDED_NAME = 'tinytom/tinytom_discarded'
LOG_NAME = 'tinytom/tinytom_settings'
LOG_DISCARDED_NAME = 'tinytom/tinytom_discarded_settings'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4-0613', help='model name')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
parser.add_argument('--max_tokens', type=int, default=450, help='max tokens')
parser.add_argument('--num', type=int, default=1, help='number of stories to generate')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--no_print', action='store_true', help='when enabled, do not print anything except progress until the end')
parser.add_argument('--simple_objects', action='store_true', default=True, help='when enabled, use the shorted object list, and store in tinytom/v3')

# tinytom generation final method --> INCLUDE three words, INCLUDE object states, DON'T INCLUDE features
# parser.add_argument('--features', action='store_true', default=False, help='whether or not to add features constraint to stories instruction')
# parser.add_argument('--three_words', action='store_true', default=True, help='whether to force 3 words from vocab into instructions. if false, use 1 word')
# parser.add_argument('--object_states', action='store_true', default=True, help='whether to force diversity in eval dataset using object state change specification')

def get_llm(args):
    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout=180
    )
    return llm

def get_human_message1(args):
    noun, verb, adj, word, features, letter = "", "", "", "", [], random.choice(letters)

    with(open(f'{PROMPT_DIR}/human_message1.txt', 'r')) as f:
        msg = f.read().strip()

    # random letter for name
    msg = msg.replace("[letter]", letter)

    # random noun
    with open(f'{WORDS_DIR}/nouns.txt', 'r') as f_noun:
        nouns = ast.literal_eval(f_noun.readline())
        noun = random.choice(nouns)
        msg = msg.replace("[noun]", noun)
    # random verb
    with open(f'{WORDS_DIR}/verbs.txt', 'r') as f_verb:
        verbs = ast.literal_eval(f_verb.readline())
        verb = random.choice(verbs)
        msg = msg.replace("[verb]", verb)
    # random adj
    with open(f'{WORDS_DIR}/adj.txt', 'r') as f_adj:
        adjs = ast.literal_eval(f_adj.readline())
        adj = random.choice(adjs)
        msg = msg.replace("[adj]", adj)

   #object states
    with open(f'{DATA_DIR}/{OBJECT_STATES_CSV}', "r") as f:
        states = f.readlines()
    prop = random.choice(states).strip().lower()
    prop = prop[0:prop.index(";")]
    msg = msg.replace('[object_property]', prop)

    if args.verbose: print(msg)
    return msg, {"noun": noun, "verb": verb, "adj": adj, "word": word, "features": features, "property": prop}

def gen_chat(args):
    response_template = """Here is the story:
Story: {story}
Reason for lack of perceptual access: {reason}
Aware of event: {awarenes}
Not aware of event: {not_aware}
Action given new state: {action_new}
Action given initial state: {action_init}
Belief Question: {belief_question}
Desire Question: {desire_question}
Action Question: {action_question}
Belief Aware: {belief_answer_aware}
Desire Aware: {desire_answer_aware}
Action Aware: {action_answer_aware}
Belief not Aware: {belief_answer_not_aware}
Desire not Aware: {desire_answer_not_aware}
Action not Aware: {action_answer_not_aware}
Random Event: {random_event}
Aware of random event: {aware_of_random_event}
Not aware of random event: {not_aware_of_random_event}
Agent Name: {agent_name}
Object: {object}"""
    llm = get_llm(args)
    with(open(f'{PROMPT_DIR}/tinytom.txt', 'r')) as f:
        instruction_text = f.read()
    
    examples = []
    template_var = ["story", "awarenes", "not_aware", "action_new", "action_init", "belief_question", "desire_question", "action_question", 
                    "belief_answer_aware", "desire_answer_aware", "action_answer_aware", "belief_answer_not_aware", "desire_answer_not_aware", 
                    "action_answer_not_aware", "random_event", "aware_of_random_event", "not_aware_of_random_event", "agent_name", "object", "reason"]
    
    csv_file = f'{DATA_DIR}/{CSV_NAME}.csv'

    prompt_tokens_used = 0
    completion_tokens_used = 0

    # counter variables
    counter = 0
    total_generated = 0
    invalid = 0
    
    # run loop to generate n stories
    while counter < args.num:
        print(f'{counter+1}/{args.num}')
        system_message = SystemMessage(content=instruction_text)

        # get hand-picked example (at zeroeth position in csv)
        with open(csv_file, "r") as f:
            l = f.readline()
            params = l.split(';')
            example_story = {k: params[v].strip() for v, k in enumerate(template_var)}
            examples.append(example_story)
        if not args.simple_objects: human_message0 = HumanMessage(content='Generate a story. The name must start with N. The story should use the verb "find", the noun "door" and the adjective "good".')        
        else: human_message0 = HumanMessage(content='Generate a story. The name must start with T. The story should use the verb "cut", the noun "pasta" and the adjective "busy".')
        ai_message = AIMessage(content=response_template.format(**examples[0]))
        s, settings = get_human_message1(args)
        human_message_1 = HumanMessage(content=s)

        # 1-shot
        messages = [system_message, human_message0, ai_message, human_message_1]
        
        if args.verbose:
            print(f"------ messages ------")
            print(messages)

            
        responses = llm.generate([messages])
        prompt_tokens_used += responses.llm_output['token_usage']['prompt_tokens']
        completion_tokens_used += responses.llm_output['token_usage']['completion_tokens']
        # price = (prompt_tokens_used * 0.03 + completion_tokens_used * 0.06) / 1000.
        # update tqdm progress bar with price
        # tqdm.tqdm.write(f"Price: {price:.2f} USD, Price per story: {price/(n_story):.2f} USD")
        for g, generation in enumerate(responses.generations[0]):
            total_generated += 1

            # print story
            if args.verbose:
                print(f"------ Generated Story {total_generated+g} ------")
                print(generation.text)
                print("------------ Fin --------------")
            
            # extract template fragments
            list_var = ["Story", "Aware of event", "Not aware of event", "Action given new state", "Action given initial state", "Belief Question", "Desire Question", "Action Question",
                        "Belief Aware", "Desire Aware", "Action Aware", "Belief not Aware",
                        "Desire not Aware", "Action not Aware", "Random Event", "Aware of random event", "Not aware of random event", "Agent Name", "Object", "Reason for lack of perceptual access"]
            out_vars = get_vars_from_out(generation.text, list_var)
            data = [out_vars[k] for k in list_var]
            data += ["auto", 0]
            
            # validate story
            with open(f'{PROMPT_DIR}/template_validation.txt', 'r') as f:
                sys_msg = SystemMessage(content=f.read().strip())
            with open(f'{PROMPT_DIR}/validation_ex_human.txt', "r") as f:
                hum_msgs = f.read().split('-')
            with open(f'{PROMPT_DIR}/validation_ex_ai.txt', "r") as f:
                ai_msgs = f.read().split('-')
            story = f"{data[0]}"
            story_msg = HumanMessage(content=story)
            if not args.no_print: print(story_msg)
            messages = [sys_msg, HumanMessage(content=hum_msgs[0].strip()), AIMessage(content=ai_msgs[0].strip()), HumanMessage(content=hum_msgs[1].strip()), AIMessage(content=ai_msgs[1].strip()), HumanMessage(content=hum_msgs[2].strip()), AIMessage(content=ai_msgs[2].strip()), HumanMessage(content=hum_msgs[3].strip()), AIMessage(content=ai_msgs[3].strip()), story_msg]
            responses = llm.generate([messages])
            for g, generation in enumerate(responses.generations[0]):
                if not args.no_print: print(generation.text)
                reasoning = generation.text.split('\n')[0].split("Reasoning:")[1]
                eval = generation.text.split("Evaluation:")[1].strip().lower()
            if eval == "invalid":
                invalid += 1
                discarded_file = f'{DATA_DIR}/{DISCARDED_NAME}.csv'
                with open(discarded_file, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(data)
                    with open(f'{DATA_DIR}/{LOG_DISCARDED_NAME}.txt', 'a') as f_settings:
                        f_settings.write(str(settings) + str({"reasoning": reasoning}) + "\n")
                continue

            # increment counter
            counter += 1

            # write to csv file
            with open(f'{DATA_DIR}/{LOG_NAME}.txt', 'a') as f_settings:
                f_settings.write(str(settings) + str({"reasoning": reasoning}) + "\n")
            story_file = f'{DATA_DIR}/{CSV_NAME}.csv'
            with open(story_file, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(data)
    # log stats
    print("Total Generated:", total_generated)
    print("Invalid:", invalid)
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    if not args.no_print: print(f"Generating {args.num} stories")
    if args.verbose:
        print(args)
    if args.simple_objects:
        CSV_NAME = 'tinytom/v3/tinytom'
        OBJECT_STATES_CSV = 'tinytom/v3/object_states.csv'
        DISCARDED_NAME = 'tinytom/v3/tinytom_discarded'
        LOG_NAME = 'tinytom/v3/tinytom_settings'
        LOG_DISCARDED_NAME = 'tinytom/v3/tinytom_discarded_settings'
    gen_chat(args)