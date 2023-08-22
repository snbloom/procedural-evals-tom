# This script is used to generate stories in a fully automatic way.
num_stories=95
num_convert=200

python ../src/tinytom.py --num $num_stories
python ../src/generate_conditions_converted.py --num $num_convert
python ../src/generate_conditions_unconverted.py