import csv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


NUM = 50
TOM_DIR = "../../data/conditions/tinytom-v3"
CONDITIONS =  ["0_forward_belief_true_belief",
                "0_forward_belief_false_belief",
                "1_forward_belief_true_belief",
                "1_forward_belief_false_belief"]
STORY_FILE = "stories.csv"
CONVERTED_STORY_FILE = "converted.txt"
CORRECTED_STORY_FILE = "corrected.txt"

def get_llm():
    llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.0,
        max_tokens=100,
        n=1,
        request_timeout=180
    )
    return llm

system = "Convert the sentences so that they can be used as queries to a language model. Make sure that clues to the state of the object are not included. These are not questions, they are incomplete sentences. If the sentence includes \"has\", convert it to is or are."

# has become , misaligned, butten, chewed cup, has been taken by.
user_1 =  "Nala believes the door to her room is unlocked."
assistant_1 = "Nala believes the door to her room is"
user_2 = "Tim believes the pasta box is not full."
assistant_2 = "Tim believes the pasta box is"
user_3 = "Lenny believes the subway tracks were misaligned but are now fixed."
assistant_3 = "Lenny believes the subway tracks are"

prompt = [SystemMessage(content=system), HumanMessage(content=user_1), AIMessage(content=assistant_1), HumanMessage(content=user_2), AIMessage(content=assistant_2),
          HumanMessage(content=user_3), AIMessage(content=assistant_3)]
llm = get_llm()

# read files
cond = CONDITIONS[0]
final_sentences = []
with open(f"{TOM_DIR}/{cond}/{STORY_FILE}", "r") as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        final_sentences.append(row[2])
converted_sentences = []
for s in final_sentences[:NUM]:
    user_n = AIMessage(content=s.strip())
    chat = prompt + [user_n]
    response = llm(chat)
    # remove newlines and spaces
    print(s)
    print(response.content)
    response = response.content.replace("\n", "").strip()
    converted_sentences.append(response)
    
# write files
for cond in CONDITIONS:
    converted_stories = []
    with open(f"{TOM_DIR}/{cond}/{CONVERTED_STORY_FILE}", "r") as f:
        stories = f.readlines()
    for i, line in enumerate(stories[:NUM]):
        text = line.strip()
        last_period_index = text.strip().rfind(".")
        story = text[:last_period_index+1]
        story = story + " " + converted_sentences[i]
        converted_stories.append(story) 
    with open(f"{TOM_DIR}/{cond}/{CORRECTED_STORY_FILE}", "w") as f:
        for story in converted_stories:
            f.write(story + "\n")