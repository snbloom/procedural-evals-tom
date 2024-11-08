import os
import csv
import json
import random


def get_tiny_tom(config, ending, final=False):
    print("ending:", ending)
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
                final_sentence = final_sentences[i]
                if ending=="think" and 'believe' in final_sentence:
                    final_sentence = final_sentence.replace('believe', 'think')
                    story = story.replace('believe', 'think')
                elif ending=="believe" and 'think' in final_sentence:
                    final_sentence = final_sentence.replace('think', 'believe')
                    story = story.replace('think', 'believe')
                elif ending=="dax":
                    final_sentence = final_sentence.replace('think', 'dax')
                    story = story.replace('think', 'dax')
                    final_sentence = final_sentence.replace('believe', 'dax')
                    story = story.replace('believe', 'dax')
                else:
                    raise Exception("Unexpected ending. Expected [think, believe, dax]")

                if not final:
                    story = story + " " + final_sentence
                else:
                    story = final_sentence
                tinytom[cond].append(story)
    # TODO: ensure all conditions are shuffled in the same order
    # keeping unshuffled for now
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
