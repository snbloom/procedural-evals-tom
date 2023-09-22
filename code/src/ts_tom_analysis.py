import re
from tqdm import tqdm

data_file = "/scr/kanishkg/TinyStories/TinyStories-train.txt"

def custom_reader(file_path):
    # Open the file at the specified path
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire content of the file
        text = file.read()
        # Split the text by the |endoftext| delimiter to create individual examples
        examples = text.split('<|endoftext|>')
        dataset = []
        if examples[-1] == '':
            examples = examples[:-1]
        for example in examples:
            dataset.append(example.strip())
        # Return all examples except the last one if it's empty (this may happen if the file ends with |endoftext|)
        return dataset
# Open the file and read its content
stories = custom_reader(data_file)

# Define the regex pattern
pattern = re.compile(r'\b(think|thinks|thought|believe|believes|believed)\b\s+\w+\s+\b(is|are|were|was)\b', re.IGNORECASE)
# pattern = re.compile(r'\b(thinks|thought|believes|believed)\b\s+(\w+\s+)?\w+\s+\b(is|are|were|was)\b', re.IGNORECASE)


num_tom_stories = 0
num_tom_sentences = 0
sentences = []
stories_tom = []

# Iterate over each line (story) in the file and find sentences that match the pattern
for story in tqdm(stories):
    tom_detected = False
    for sentence in story.split('.'):
        matches = pattern.findall(sentence)
        if matches:
            tom_detected = True
            num_tom_sentences += 1
            sentences.append(sentence.replace('\n',' ')+'\n')
    if tom_detected:
        num_tom_stories += 1
        stories_tom.append(story.replace('\n',' ')+'\n')

print("Number of stories with Tom: ", num_tom_stories)
print("Number of sentences with Tom: ", num_tom_sentences)

with open('/scr/kanishkg/TinyStories/ts-tom.txt','w') as f:
    f.writelines(stories_tom[:20])
with open('/scr/kanishkg/TinyStories/ts-tom-sentences.txt','w') as f:
    f.writelines(sentences[:100])
