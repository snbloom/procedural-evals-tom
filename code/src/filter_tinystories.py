from tqdm import tqdm
import os
import json

TINYSTORIES = "/scr/kanishkg/TinyStories/"
TINYSTORIES_V2 = "/scr/kanishkg/TinyStories/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_NO_THINK_BELIEVE = "/scr/snbloom/TinyStories_No_Think_Believe/TinyStories_No_Think_Believe_gpt4.txt"

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
    with open(os.path.join(TINYSTORIES_V2), 'r') as f:
        strs = f.read().split("<|endoftext|>")
        print(len(strs))
        for s in tqdm(strs):
            s = s.strip()
            print(s.replace('\n\n', '\n').replace('\n', " "))
            if no_think_believe and ("think" in s or "believe" in s): continue
            else: stories.append(s.replace('\n\n', '\n').replace('\n', " "))
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

with open(TINYSTORIES_NO_THINK_BELIEVE, "w") as f:
    for story in stories:
        f.write(story+'\n')

print(f"Number of filtered stories: {len(filtered)}")
print(f"Number of stories swapped: {num_replaced}")