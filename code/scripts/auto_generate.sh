# This script is used to generate stories in a fully automatic way.
python ../src/tinytom.py --num $1
python ../src/generate_conditions_converted.py
python ../src/generate_conditions_unconverted.py