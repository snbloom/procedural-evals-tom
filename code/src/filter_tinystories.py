from tqdm import tqdm
import os
import json

TINYSTORIES = "/scr/kanishkg/TinyStories/"
TINYSTORIES_V2 = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_NO_THINK_BELIEVE_FOLDER = "/scr/snbloom/TinyStories_No_Think_Believe"
TINYSTORIES_NO_THINK_BELIEVE_NAME = "TinyStories_No_Think_Believe_gpt4.txt"

def get_tiny_stories():
    stories = []
    for i in range(1):
        filename = f'data{i:02}.json'
        with open(os.path.join(TINYSTORIES, filename), 'r') as f:
            data = json.load(f)
            for d in tqdm(data):
                # only select stories from gpt-4
                if d['source'] != 'GPT-4':
                    continue
                stories.append(d['story'].strip())
    print(f"Number of tinystories stories: {len(stories)}")
    return stories

def get_tiny_stories_v2(no_think_believe=True):
    stories = []
    most_recent = ""
    with open(os.path.join(TINYSTORIES_V2), 'r') as f:
        for line in tqdm(f):
            if line.strip() != "<|endoftext|>": most_recent += line.strip().replace('\n', "")
            else:
                if no_think_believe and ("think" in most_recent or "believe" in most_recent): 
                    print("THINK OR BELIEVE")
                else:
                    most_recent = most_recent.replace('\n', "")
                    print(most_recent)
                    stories.append(most_recent)
                most_recent = ""
    print(f"Number of tinystories_v2 stories: {len(stories)}")
    return stories

filtered = []
num_replaced = 0

print("Parsing tiny stories")
stories = get_tiny_stories()
print("Parsing tiny stories v2")
stories_v2 = get_tiny_stories_v2()

for story in tqdm(stories):
    if "think" in story or "believe" in story: 
        # add story from v2 set
        filtered.append(stories_v2[num_replaced])
        num_replaced += 1
        continue
    else: filtered.append(story)

if not os.path.exists(TINYSTORIES_NO_THINK_BELIEVE_FOLDER):
    os.makedirs(TINYSTORIES_NO_THINK_BELIEVE_FOLDER)
with open(os.path.join(TINYSTORIES_NO_THINK_BELIEVE_FOLDER, TINYSTORIES_NO_THINK_BELIEVE_NAME), "w") as f:
    for story in stories:
        f.write(story+'\n')

print(f"Number of filtered stories: {len(filtered)}")
print(f"Number of stories swapped: {num_replaced}")