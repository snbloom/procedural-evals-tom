# evaluate conditions for model in arguments
python ../src/auto_eval.py --condition true_belief --init_belief 0_forward --model_id finetuned-28-v3-600 --local --unconverted
python ../src/auto_eval.py --condition false_belief --init_belief 0_forward --model_id finetuned-28-v3-600 --local --unconverted