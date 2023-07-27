import os
import csv
import json
import random


def get_tiny_tom(config):
    tinytom = {}
    for cond in config["conditions"]:
        tinytom[cond] = []
        final_sentences = []
        with open(os.path.join(config["tom_data_dir"], cond, config["condition_file"]), 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                final_sentences.append(row[2])
        with open(os.path.join(config["tom_data_dir"], cond, config["story_file"]), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                text = line.strip()
                last_period_index = text.strip().rfind(".")
                story = text[:last_period_index+1]
                story = story + " " + final_sentences[i]
                tinytom[cond].append(story)
        random.shuffle(tinytom[cond])
    return tinytom

def get_tiny_stories(config, count):
    stories = []
    for i in range(1):
        filename = f'data{i:02}.json'
        with open(os.path.join(config["tinystories_data_dir"], filename), 'r') as f:
            data = json.load(f)
            for d in data:
                # only select stories from gpt-4
                if d['source'] != 'GPT-4':
                    continue
                stories.append(d['story'].strip())
    print(f"Number of tinystories stories: {len(stories)}, keeping {count}")
    stories = random.sample(stories, count)
    return stories
