import os
import json
from tqdm import tqdm
import argparse

# load data from hf datasets
TS_DIR = "/scr/kanishkg/TinyStories/"
TS_DIR_OUT = "/scr/kanishkg/TinyStories/"
TINYSTORIES_V2 = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_V2_VAL = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-valid.txt"
train_file = os.path.join(TS_DIR, "TinyStories-train.txt")
val_file = os.path.join(TS_DIR, "TinyStories-valid.txt")

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--target_words', type=str, default="think_believe", help='[think_believe, know, think_and_know, learn, feel, prefer, want, plan, time]')

target_words = []

think_words = ['think', "believe", "thought", "belief"]
know_words = ["know", "knew"]
think_and_know_words = think_words + know_words
learn_words = ["learn", "didn't know", "did not know", "realize", "teach", "taught", "something new", "a new thing", "never seen before", "ask", "help"]
feel_words = ["feel", "felt"]
prefer_words = ["like", "love", "prefer"]
want_words = ["want", "wish"]
plan_words = ["plan", "decide"]
time_words = ["now", "lately", "earlier", "soon", "currently", "today", "tomorrow", "yesterday", "soon", "later", "always", "never", "forever", "before", "after", "during", "while", "when", "then", "suddenly", "frequently", "rarely", "sometimes", "often", "currently", "recently", "prior to", "subsequently", "simultaneously", "eventually", "in the meantime", "afterward", "previously"]

args = parser.parse_args()
if args.target_words == "think_believe": target_words = think_words
elif args.target_words == "know": target_words = know_words
elif args.target_words == "think_and_know": target_words = think_and_know_words
# elif args.target_words == "learn": target_words = learn_words
# elif args.target_words == "feel": target_words = feel_words
# elif args.target_words == "prefer": target_words = prefer_words
# elif args.target_words == "want": target_words = want_words
# elif args.target_words == "plan": target_words = plan_words
# elif args.target_words == "time": target_words = time_words
else: raise Exception("Unexpected target_words type. Expected: [think_believe, know, learn, feel, prefer, want, plan, time]")

def has_no_target_words(text):
    for word in target_words:
        if word.lower() in text.lower(): return False
    return True

def has_target_words(text):
    for word in target_words:
        if word in text: return True
    return False

def get_tiny_stories_v2(train=True):
    stories = []
    most_recent = ""
    if train: path = TINYSTORIES_V2
    else: path = TINYSTORIES_V2_VAL

    with open(os.path.join(path), 'r') as f:
        for line in tqdm(f):
            if line.strip() != "<|endoftext|>": most_recent += line.strip().replace('\n', "")
            else:
                most_recent = most_recent.replace('\n', "")
                stories.append(most_recent)
                most_recent = ""
    print(f"Number of tinystories_v2 stories: {len(stories)}")
    return stories

def custom_reader(file_path, tinystories_v2):

    # Open the file at the specified path
    with open(file_path, 'r', encoding='utf-8') as file:
        
        dataset = []
        num_replaced = 0
        replacement_idx = 0
        most_recent = ""

        for line in tqdm(file):
            # if not end of text label, add this line to the most recent story
            if line.strip() != "<|endoftext|>": most_recent += line.strip().replace('\n', "")
            # otherwise, this is the end of the story
            else:
                # append the story to the dataset if has the target words
                if has_target_words(most_recent): dataset.append({"text": most_recent})
                # if doesn't have target words, replace with a story from the other dataset (with target words)
                else: 
                    num_replaced += 1
                    replacement = tinystories_v2[replacement_idx] 
                    while has_no_target_words(replacement):
                        replacement_idx += 1
                        if replacement_idx == len(tinystories_v2)-1: print("ERROR: replacement_idx at length of v2 stories")
                        replacement = tinystories_v2[replacement_idx] 
                    replacement_idx += 1
                    dataset.append({"text": replacement})
                # clear string tracker for next story
                most_recent = ""
        print(f"Number tinystories swapped: {num_replaced}")
        print(f"Number of v2 tinystories tried: {replacement_idx}")
        print(f"Number of v2 tinystories tried without target words: {replacement_idx - num_replaced}")
        return dataset

print()
print(f"TRAIN AND VAL FILTERING FOR CONDITION: {args.target_words.upper()}")
print("Getting stories v2 train")
tinystories_v2 = get_tiny_stories_v2()
print()
print("Getting stories v2 valid")
tinystories_v2_val = get_tiny_stories_v2(train=False)
print()

print("Reading in train file")
train_ex = custom_reader(train_file, tinystories_v2)
print(f"Train length: {len(train_ex)}")
print()

print("Reading in val file")
val_ex = custom_reader(val_file, tinystories_v2_val)
print(f"Val length: {len(val_ex)}")
print()

def store_json(path, data_dict):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_dict:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

if not os.path.exists(TS_DIR_OUT):
    os.makedirs(TS_DIR_OUT)

print(f"Writing to json files: {TS_DIR_OUT}train_no_{args.target_words}.json and {TS_DIR_OUT}val_no_{args.target_words}.json")
store_json(TS_DIR_OUT+f'train_no_{args.target_words}.json', train_ex)
store_json(TS_DIR_OUT+f'val_no_{args.target_words}.json', val_ex)

with open(TS_DIR_OUT+f'train_no_{args.target_words}.json', 'r') as f:
    print(f.read()[:20000])