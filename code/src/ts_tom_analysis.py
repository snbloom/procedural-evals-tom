import re
from tqdm import tqdm

data_file = "/scr/kanishkg/TinyStories/TinyStories-train.txt"

# Open the file and read its content
with open(data_file, 'r') as file:
    stories = file.readlines()

# Define the regex pattern
pattern = re.compile(r'\b(thinks|thought|believes|believed)\b\s+\w+\s+\b(is|are|were|was)\b', re.IGNORECASE)

num_tom_stories = 0
num_tom_sentences = 0

# Iterate over each line (story) in the file and find sentences that match the pattern
for story in tqdm(stories):
    tom_detected = False
    for sentence in story.split('.'):
        matches = pattern.findall(sentence)
        if matches:
            tom_detected = True
            num_tom_sentences += 1
    if tom_detected:
        num_tom_stories += 1

print("Number of stories with Tom: ", num_tom_stories)
print("Number of sentences with Tom: ", num_tom_sentences)
