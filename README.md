##  

A Domain-Agnostic Method for Procedurally Generating LLM Evaluations

![Causal Template -> Prompt Template -> Test Items](./assets/generation.jpg)


### 🧐 What is this?
This is a supporting repository for a project based off our lab's prior paper titled "Understanding Social Reasoning in LLMs with LLMs".

Prior project: we develop a method that uses large language models (LLMs) to procedurally generate evaluations for other LLMs. We apply this method to assess the performance of LLMs in a subdomain of social reasoning (Theory-of-Mind). Please checkout our [paper](https://sites.google.com/view/social-reasoning-lms) for further details.

This project: we adapt the BigToM generation and evaluation method for models trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. 

### 📂 Repo structure
```
├── code                 
│   └── analysis
│   └── prolific-exp-1
│   └── prolific-exp-2
│   └── prompt_instructions
│   └── scripts
│   └── src 
├── data   
│   ├── bigtom    
│   └── expert_data
│   └── results
│   └── social_iqa
│   └── tinytom
│   └── training
├── .gitignore
├── LICENSE            
└── requirements.txt
```

### 🚀 Getting started 
#### Using miniconda
1. `curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh`
2. `bash Miniconda3-latest-MacOSX-x86_64.sh`
3. close and reopen terminal
4. `source ~/.bashrc` or `source ~/.zshrc`
5. `conda create --name tinytom python==3.10`
6. `conda activate tinytom`
7. `pip install -r requirements.txt` 

#### Generating TinyToM
Prompt for generating TinyToM is in `code/prompt_instructions/tinytom.txt` and the python script is at `code/src/tinytom.py`. To generate TinyToM, run the following commands:
1. `cd code/src`
2. `python tinytom.py` (use `python tinytom.py --num_stories [NUM_STORIES]` to specify number of stories to generate)
3. `python generate_conditions.py`
4. `python generate_tinytom_stories.py`

#### Evaluating on TinyToM
We provide code to evaluate models on TinyToM in `code/src/auto_eval.py`. More specific experiment scripts are available in `code/scripts`.

----

#### Generating BigToM
Prompt for generating BigToM is in `code/prompt_instructions/bigtom.txt` and the python script is at `code/src/bigtom.py`. To generate BigToM, run the following commands:
1. `cd code/src`
2. `python bigtom.py`
3. `python generate_conditions.py`

#### Human Experiments
We provide code to run Human experiments of 3 kinds:
1. Expert Ratings: `code/src/expert_evaluate.py`
2. Prolific Experiment for Rating Generated Stories: `code/prolific-exp-1`
3. Prolific Experiment for Testing Human Participants: `code/prolific-exp-2`

#### Evaluating on BigToM
We provide code to evaluate models on BigToM in `code/src/evaluate_conditions.py`. More specific experiment scripts are available in `code/scripts`.

#### Process to Generate 
1. Run tinytom.py
2. Run generate_conditions.py
3. Run generate_tinytom_stories.py
