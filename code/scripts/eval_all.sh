# evaluate conditions for model in arguments
python ../src/auto_eval.py --condition true_belief --init_belief 0_forward --model_id $1 --no_print
python ../src/auto_eval.py --condition false_belief --init_belief 0_forward --model_id $1 --no_print
python ../src/auto_eval.py --condition true_belief --init_belief 1_forward --model_id $1 --no_print
python ../src/auto_eval.py --condition false_belief --init_belief 1_forward --model_id $1 --no_print