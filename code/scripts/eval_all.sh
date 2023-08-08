# positive controls
python ../src/auto_eval.py --model_id 28M --init_belief 0_forward --condition true_belief --num 10 --local
python ../src/auto_eval.py --model_id 28M --init_belief 0_forward --condition false_belief --num 10 --local
python ../src/auto_eval.py --model_id 28M --init_belief 1_forward --condition true_belief --num 10 --local
python ../src/auto_eval.py --model_id 28M --init_belief 1_forward --condition false_belief --num 10 --local
python ../src/auto_eval.py --model_id 33M --init_belief 0_forward --condition true_belief --num 10 --local
python ../src/auto_eval.py --model_id 33M --init_belief 0_forward --condition false_belief --num 10 --local
python ../src/auto_eval.py --model_id 33M --init_belief 1_forward --condition true_belief --num 10 --local
python ../src/auto_eval.py --model_id 33M --init_belief 1_forward --condition false_belief --num 10 --local

# unconverted versions
python ../src/auto_eval.py --model_id gpt35turbo --init_belief 0_forward --condition true_belief --unconverted
python ../src/auto_eval.py --model_id gpt35turbo --init_belief 0_forward --condition false_belief --unconverted
python ../src/auto_eval.py --model_id gpt35turbo --init_belief 1_forward --condition true_belief --unconverted
python ../src/auto_eval.py --model_id gpt35turbo --init_belief 1_forward --condition false_belief --unconverted
python ../src/auto_eval.py --model_id davinci003 --init_belief 0_forward --condition true_belief --unconverted
python ../src/auto_eval.py --model_id davinci003 --init_belief 0_forward --condition false_belief --unconverted
python ../src/auto_eval.py --model_id davinci003 --init_belief 1_forward --condition true_belief --unconverted
python ../src/auto_eval.py --model_id davinci003 --init_belief 1_forward --condition false_belief --unconverted
