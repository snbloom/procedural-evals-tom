import os
import csv
import argparse


DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions/bigtom')
CSV_NAME = os.path.join(DATA_DIR, 'bigtom/bigtom.csv')
CONVERTED_CSV_NAME = os.path.join(DATA_DIR, 'conditions/bigtom/0_forward_action_false_belief/stories_trimmed.csv')

revised_stories = []

def get_stories(csv_name):
    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=";")
        stories = list(reader)
    return stories


def generate_conditions(stories, trimmed):
    list_var = ["Story", "Aware of event", "Not Aware of event", "Action aware", "Action not aware", "Belief Question", "Desire Question", "Action Question",
                "Belief Answer Aware", "Desire Answer Aware", "Action Answer Aware", "Belief Answer not Aware",
                "Desire Answer not Aware", "Action Answer not Aware", "Random Event", "Aware of random event", "Not aware of random event", "Agent name", "Object"]

    for i, story in enumerate(stories):
        trimmed_story = trimmed[i]
        story = story[:-2]
        print(story)
        print(trimmed_story)

        name = trimmed_story[0].split()[0]
        obj = trimmed_story[1]

        print(obj)
        print(name)

        story.append(name)
        story.append(obj)

        revised_stories.append(story)

        print(story)

    print(revised_stories)
        
    # write to file
    with open(CSV_NAME, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for revised in revised_stories:
            revised += ["auto", 0]
            writer.writerow(revised)



if __name__ == "__main__":  
    stories = get_stories(CSV_NAME)
    trimmed = get_stories(CONVERTED_CSV_NAME)
    generate_conditions(stories, trimmed)
