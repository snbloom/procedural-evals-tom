import os
import argparse
import random
import ast
from tqdm import tqdm
import csv

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

MODEL = "gpt-3.5-turbo-16k-0613"
DATA_DIR = '../../data'
TRAINING_DIR = DATA_DIR + "/training"
PROMPT_DIR = '../prompt_instructions'
INSTRUCTIONS = PROMPT_DIR + "/tinytom_training_stories.txt"
WORDS_DIR = '../tinystories_words'


ATTRIBUTES = ["beliefs", "desires", "feelings", "morals"]
METHODS = ["change", "conflicting", "expresses", "interacts"]
METHODS_TO_SENTENCE = {
    "change": "something happens which causes an agent to change their [attribute]",
    "conflicting": "two characters have conflicting [attribute]",
    "expresses": "a character directly expresses their [attribute]",
    "interacts": "one character interacts with an object in a way that shows their [attribute]"
}

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']

parser = argparse.ArgumentParser()
parser.add_argument('--num_desire', type=int, default="10", help="how many desire training stories")
parser.add_argument('--num_belief', type=int, default="10", help="how many belief training stories")
parser.add_argument('--num_morality', type=int, default="10", help="how many morality training stories")
parser.add_argument('--num_feeling', type=int, default="10", help="how many feeling training stories")

parser.add_argument('--num_each', type=int, default="10", help="how many training stories of each")


def get_llm():
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.0,
        max_tokens=450,
        n=1,
        request_timeout=180
    )
    return llm

def get_formatted_instructions(features):
    noun, verb, adj, letter = "", "", "", random.choice(letters)
    with(open(INSTRUCTIONS, 'r')) as f:
        msg = f.read().strip()

    # letter
    msg = msg.replace("[letter]", letter)

    # features
    msg = msg.replace("[features]", ",".join(features))

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

    print(msg)
    return {"noun": noun, "verb": verb, "adj": adj, "instructions": msg}

def generate_stories(args):
    # Attributes:
        # Beliefs
        # Desires
        # Feelings 

    # Methods:
        # Two agents have conflicting [attribute] 
        # An agent directly expresses their [attribute]
        # One agent interacts with an object in a way that shows their [attribute] 
        # Something happens which causes an agent to change their [attribute]

    for _ in tqdm(range(args.num_each)):
        # general unawareness of something in the environment
        features = ["the characters in the story are unaware of something in their environment"]

        llm = get_llm()
        msg = get_formatted_instructions(features)
        system_message = SystemMessage(content=msg["instructions"])
        responses = llm.generate([[system_message]])
        for g, generation in enumerate(responses.generations[0]):
            print(generation.text.replace('\n', " "))

            with open(TRAINING_DIR + "/awareness.csv", 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow([generation.text.replace('\n', " "), msg])

        # just beliefs for now
        attribute = "beliefs"

        for method in METHODS:
            features = [METHODS_TO_SENTENCE[method].replace("[attribute]", attribute)]
            msg = get_formatted_instructions(features)
            system_message = SystemMessage(content=msg["instructions"])
            responses = llm.generate([[system_message]])
            for g, generation in enumerate(responses.generations[0]):
                print(generation.text.replace('\n', " "))

                with open(TRAINING_DIR + f"/{attribute}/{attribute}_{method}.csv", 'a') as csvfile:
                        writer = csv.writer(csvfile, delimiter=';')
                        writer.writerow([generation.text.replace('\n', " "), msg])

if __name__ == "__main__":  
    args = parser.parse_args()
    generate_stories(args)