import re
import json
from tqdm import tqdm

# data_file = "/scr/kanishkg/TinyStories/TinyStories-train.txt"
data_file = "/scr/kanishkg/TinyStories/train_appraisal.json"

keep_appraisal = True
appraisal_list = [
        "fun", "stupid", "interesting", "smart", "amazing", "awesome", "brilliant", "nice", "sweet", "best", "pretty","special", "good", "cool", "great", "crazy", "bad", "boring", "exciting", "surprising", "disgusting", "silly",
            "thrilling", "unimpressive", "fantastic", "mediocre", "impressive", "dull", 
                "enjoyable", "tedious", "lovely", "disturbing", "refreshing", "unpleasant", 
                    "delightful", "monotonous", "stunning", "horrible", "mesmerizing", "annoying",
                        "charming", "pathetic", "beautiful", "ugly", "invigorating", "troubling", 
                            "gorgeous", "uninspiring", "lively", "dreary", "innovative", "outdated",
                                "magnificent", "depressing", "spectacular", "lackluster", "intriguing", "tedious"
                                ]


def custom_reader(file_path):
    # Open the file at the specified path
    if "txt" in file_path:
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
    elif "json" in file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_list=list(file)
        dataset = []
        for json_str in json_list:
            story = json.loads(json_str)
            dataset.append(story["text"])
    # Return all examples except the last one if it's empty (this may happen if the file ends with |endoftext|)
    return dataset
# Open the file and read its content
stories = custom_reader(data_file)

# Define the regex pattern
pattern = re.compile(r'\b(think|thinks|thought|believe|believes|believed)\b\s+\w+\s+\b(is|are|were|was)\b', re.IGNORECASE)

# pattern = re.compile(r'\b(expects|expected)\b\s+(\w+\s+)?\w+\s+\b(to|is|are|were|was)\b', re.IGNORECASE)
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
            if keep_appraisal:
                for w in appraisal_list:
                    if w in sentence:
                        matches = False
                        break
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
    f.writelines(sentences[:200])
