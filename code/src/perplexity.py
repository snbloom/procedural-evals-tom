import evaluate

DATA_DIR = '../../data/tinytom/tinytom_converted_parts.txt'
contexts = []

with open(DATA_DIR, 'r') as f:
    for story in list(f.readlines()):
        contexts.append(story[0])

for i, context in enumerate(contexts):
    print(context)
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_texts = [context]
    results = perplexity.compute(model_id='roneneldan/TinyStories-28M',
                                add_start_token=True,
                                predictions=input_texts)
    print(list(results.keys()))

    if i > 1: break