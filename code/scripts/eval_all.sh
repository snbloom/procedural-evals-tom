# evaluate conditions for model in arguments
python ../src/auto_eval.py --condition true_belief --init_belief 0_forward --model_id $1 --local --data_dir $2
python ../src/auto_eval.py --condition false_belief --init_belief 0_forward --model_id $1 --local --data_dir $2
python ../src/auto_eval.py --condition true_belief --init_belief 1_forward --model_id $1 --local --data_dir $2
python ../src/auto_eval.py --condition false_belief --init_belief 1_forward --model_id $1 --local --data_dir $2