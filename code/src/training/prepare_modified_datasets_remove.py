import os
import re
import json
from tqdm import tqdm
import argparse

from prepare_gpt4 import custom_reader

# load data from hf datasets
TS_DIR = "/scr/kanishkg/TinyStories/"
TS_DIR_OUT = "/scr/kanishkg/TinyStories/"
TS_V2 = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-train.txt"
TS_V2_VAL = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-valid.txt"
train_file = os.path.join(TS_DIR, "TinyStories-train.txt")
val_file = os.path.join(TS_DIR, "TinyStories-valid.txt")

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--method', default='banned_words', help='[banned_words, regex]')
parser.add_argument('--banned_words', type=str, default="think_believe", help='[think_believe, know, think_and_know, learn, feel, prefer, want, plan, time]')

banned_words = []

think_words = ['think', "believe", "thought", "belief"]
know_words = ["know", "knew"]
think_and_know_words = think_words + know_words
learn_words = ["learn", "didn't know", "did not know", "realize", "teach", "taught", "something new", "a new thing", "never seen before", "ask", "help"]
feel_words = ["feel", "felt"]
prefer_words = ["like", "love", "prefer"]
want_words = ["want", "wish"]
plan_words = ["plan", "decide"]
time_words = ["now", "lately", "earlier", "soon", "currently", "today", "tomorrow", "yesterday", "soon", "later", "always", "never", "forever", "before", "after", "during", "while", "when", "then", "suddenly", "frequently", "rarely", "sometimes", "often", "currently", "recently", "prior to", "subsequently", "simultaneously", "eventually", "in the meantime", "afterward", "previously"]

appraisal_list = [
        "fun", "stupid", "interesting", "smart", "amazing", "awesome", "brilliant",
        "nice", "sweet", "best", "pretty","special", "good", "cool", "great", "crazy",
        "bad", "boring", "exciting", "surprising", "disgusting", "silly",
        "thrilling", "unimpressive", "fantastic", "mediocre", "impressive", "dull", 
        "enjoyable", "tedious", "lovely", "disturbing", "refreshing", "unpleasant", 
        "delightful", "monotonous", "stunning", "horrible", "mesmerizing", "annoying",
        "charming", "pathetic", "beautiful", "ugly", "invigorating", "troubling", 
        "gorgeous", "uninspiring", "lively", "dreary", "innovative",
        "magnificent", "depressing", "spectacular", "lackluster", "intriguing", "tedious"
                ]


args = parser.parse_args()
if args.banned_words == "think_believe": banned_words = think_words
elif args.banned_words == "know": banned_words = know_words
elif args.banned_words == "think_and_know": banned_words = think_and_know_words
elif args.banned_words == "learn": banned_words = learn_words
elif args.banned_words == "feel": banned_words = feel_words
elif args.banned_words == "prefer": banned_words = prefer_words
elif args.banned_words == "want": banned_words = want_words
elif args.banned_words == "plan": banned_words = plan_words
elif args.banned_words == "time": banned_words = time_words
else: raise Exception("Unexpected banned_words type. Expected: [think_believe, know, learn, feel, prefer, want, plan, time]")

def has_banned_words(text):
    for word in banned_words:
        if word.lower() in text.lower(): return True
    return False 

def has_pattern(story, keep_appraisal=True):
    pattern = re.compile(r'\b(think|thinks|thought|believe|believes|believed)\b\s+\w+\s+\b(is|are|were|was)\b', re.IGNORECASE)
    tom_detected = False
    for sentence in story.split('.'):
        matches = pattern.findall(sentence)
        if matches:
            if keep_appraisal:
                for w in appraisal_list:
                    if w in sentence:
                        matches = False
                        break
        if matches:
            return True
    return False 

def filter_and_replace(ts, ts_v2):

    dataset = []
    num_replaced = 0
    replacement_idx = 0

    for story in tqdm(ts):
        # append the story to the dataset if no banned words
        if (args.method == "banned_words" and not has_banned_words(story["text"])) or (args.method == "regex" and not has_pattern(story["text"])) :
            dataset.append(story)
        # if has banned words, replace with a story from the other dataset (with no banned words)
        else: 
            num_replaced += 1
            replacement = ts_v2[replacement_idx] 
            while (args.method == "banned_words" and has_banned_words(replacement["text"])) or (args.method == "regex" and has_pattern(replacement["text"])):
                replacement_idx += 1
                if replacement_idx == len(ts_v2)-1: print("ERROR: replacement_idx at length of v2 stories")
                replacement = ts_v2[replacement_idx] 
            replacement_idx += 1
            dataset.append(replacement)

    print(f"Number tinystories swapped: {num_replaced}")
    print(f"Number of v2 tinystories tried: {replacement_idx}")
    print(f"Number of v2 tinystories with banned words: {replacement_idx - num_replaced}")
    return dataset


train_ex = custom_reader(train_file)
print(f"v1 train stories: {len(train_ex)}")
val_ex = custom_reader(val_file)
print(f"v1 val stories: {len(val_ex)}")

v2_train = custom_reader(TS_V2)
print(f"v2 train stories: {len(v2_train)}")
v2_val = custom_reader(TS_V2_VAL)
print(f"v2 val stories: {len(v2_val)}")

print(f"Filtering and replacing banned words for train: {banned_words}")
new_train = filter_and_replace(train_ex, v2_train)
print(f"Filtering and replacing banned words for val: {banned_words}")
new_val = filter_and_replace(val_ex, v2_val)

def store_json(path, data_dict):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_dict:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

if not os.path.exists(TS_DIR_OUT):
    os.makedirs(TS_DIR_OUT)

print(f"Writing to json files: {TS_DIR_OUT}train_no_{args.banned_words}.json and {TS_DIR_OUT}val_no_{args.banned_words}.json")
store_json(TS_DIR_OUT+f'train_{args.method}.json', new_train)
store_json(TS_DIR_OUT+f'val_{args.method}.json', new_val)

with open(TS_DIR_OUT+f'train_{args.method}.json', 'r') as f:
    print(f.read()[:2000])
