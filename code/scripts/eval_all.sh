# evaluate conditions
python ../src/evaluate_conditions.py --num 50 --condition true_belief --init_belief 0_forward --model_name gpt-3.5-turbo
python ../src/evaluate_conditions.py --num 50 --condition false_belief --init_belief 0_forward --model_name gpt-3.5-turbo
python ../src/evaluate_conditions.py --num 50 --condition true_belief --init_belief 1_forward --model_name gpt-3.5-turbo
python ../src/evaluate_conditions.py --num 50 --condition false_belief --init_belief 1_forward --model_name gpt-3.5-turbo