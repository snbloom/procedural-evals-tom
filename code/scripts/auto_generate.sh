# This script is used to generate stories in a fully automatic way.
num_stories=100
num_convert=120

python ../src/tinytom.py --num $num_stories
python ../src/generate_conditions_tinytom.py --num $num_convert