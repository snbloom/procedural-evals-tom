# evaluate conditions
# python ../src/auto_eval.py --condition true_belief --init_belief 1_forward --model_id davinci003 --bigtom
# python ../src/auto_eval.py --condition false_belief --init_belief 1_forward --model_id davinci003 --bigtom

python ../src/auto_eval.py --condition true_belief --init_belief 0_forward --model_id gpt35turbo --bigtom
python ../src/auto_eval.py --condition false_belief --init_belief 0_forward --model_id gpt35turbo --bigtom
python ../src/auto_eval.py --condition true_belief --init_belief 1_forward --model_id gpt35turbo --bigtom
python ../src/auto_eval.py --condition false_belief --init_belief 1_forward --model_id gpt35turbo --bigtom