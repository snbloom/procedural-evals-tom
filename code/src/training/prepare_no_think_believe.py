import os
import json

# load data from hf datasets
TS_DIR = "/scr/kanishkg/TinyStories/"
TS_DIR_OUT = "/scr/snbloom/TinyStories/"
TINYSTORIES_V2 = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-train.txt"
train_file = os.path.join(TS_DIR, "TinyStories-train.txt")
val_file = os.path.join(TS_DIR, "TinyStories-valid.txt")

banned_words = ['think', "believe"]

def has_no_banned_words(text, banned_words=banned_words):
    for word in banned_words:
        if word in text: return False
    return True

def get_tiny_stories_v2(banned_words=banned_words):
    stories = []
    most_recent = ""
    with open(os.path.join(TINYSTORIES_V2), 'r') as f:
        for line in f:
            if line.strip() != "<|endoftext|>": most_recent += line.strip().replace('\n', "")
            else:
                if has_no_banned_words(most_recent):
                    most_recent = most_recent.replace('\n', "")
                    stories.append(most_recent)
                most_recent = ""
    print(f"Number of tinystories_v2 stories: {len(stories)}")
    return stories

def custom_reader(file_path):
    tinystories_v2 = get_tiny_stories_v2()

    # Open the file at the specified path
    with open(file_path, 'r', encoding='utf-8') as file:
        
        dataset = []
        num_replaced = 0
        replacement_idx = 0
        most_recent = ""

        for line in file:
            # if not end of text label, add this line to the most recent story
            if line.strip() != "<|endoftext|>": most_recent += line.strip().replace('\n', "")
            # otherwise, this is the end of the story
            else:
                # append the story to the dataset if no banned words
                if has_no_banned_words(most_recent): dataset.append({"text": most_recent})
                # if has banned words, replace with a story from the other dataset (with no banned words)
                else: 
                    num_replaced += 1
                    replacement = tinystories_v2[replacement_idx] 
                    while not has_no_banned_words(replacement):
                        replacement_idx += 1
                        replacement = tinystories_v2[replacement_idx] 
                    replacement_idx += 1
                    dataset.append({"text": replacement})
                # clear string tracker for next story
                most_recent = ""

        return dataset

train_ex = custom_reader(train_file)
print(len(train_ex))
val_ex = custom_reader(val_file)
print(len(val_ex))

def store_json(path, data_dict):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_dict:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

if not os.path.exists(TS_DIR_OUT):
    os.makedirs(TS_DIR_OUT)

store_json(TS_DIR_OUT+'train_no_think_believe.json', train_ex)
store_json(TS_DIR_OUT+'val_no_think_believe.json', val_ex)

