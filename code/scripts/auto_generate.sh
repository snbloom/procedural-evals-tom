# This script is used to generate stories in a fully automatic way.
num_stories=1

python ../src/tinytom.py --num $num_stories
python ../src/generate_conditions_converted.py
python ../src/generate_conditions_unconverted.py