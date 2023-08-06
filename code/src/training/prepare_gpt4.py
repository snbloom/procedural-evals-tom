import os
import json

# load data from hf datasets
TS_DIR = "/scr/kanishkg/TinyStories/"
train_file = os.path.join(TS_DIR, "TinyStories-train.txt")
val_file = os.path.join(TS_DIR, "TinyStories-valid.txt")

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
            dataset.append({"text": example.strip()})
        # Return all examples except the last one if it's empty (this may happen if the file ends with |endoftext|)
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
store_json(TS_DIR+'train.json', train_ex)
store_json(TS_DIR+'val.json', val_ex)

