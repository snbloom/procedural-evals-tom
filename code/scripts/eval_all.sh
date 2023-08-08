# evaluate conditions
python ../src/auto_eval.py --model_id 28M --init_belief 0_forward --condition true_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 28M --init_belief 0_forward --condition false_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 28M --init_belief 1_forward --condition true_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 28M --init_belief 1_forward --condition false_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 33M --init_belief 0_forward --condition true_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 33M --init_belief 0_forward --condition false_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 33M --init_belief 1_forward --condition true_belief --num 10 --local --unconverted
python ../src/auto_eval.py --model_id 33M --init_belief 1_forward --condition false_belief --num 10 --local --unconverted
