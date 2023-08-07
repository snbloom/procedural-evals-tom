python ../src/auto_eval.py -–model_id davinci003 --init_belief 0_forward --condition true_belief
python ../src/auto_eval.py -–model_id davinci003 --init_belief 0_forward --condition false_belief
python ../src/auto_eval.py -–model_id davinci003 --init_belief 1_forward --condition true_belief
python ../src/auto_eval.py -–model_id davinci003 --init_belief 1_forward --condition false_belief

python ../src/auto_eval.py -–model_id gpt35turbo --init_belief 0_forward --condition true_belief
python ../src/auto_eval.py -–model_id gpt35turbo --init_belief 0_forward --condition false_belief
python ../src/auto_eval.py -–model_id gpt35turbo --init_belief 1_forward --condition true_belief
python ../src/auto_eval.py -–model_id gpt35turbo --init_belief 1_forward --condition false_belief