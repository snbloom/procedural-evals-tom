#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=128
#SBATCH --time=3-0
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load any necessary modules or environment variables here
# For example:



__conda_setup="$('/scr/kanishkg/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
    else
	if [ -f "/scr/kanishkg/miniconda3/etc/profile.d/conda.sh" ]; then
		. "/scr/kanishkg/miniconda3/etc/profile.d/conda.sh"
	else
		export PATH="/scr/kanishkg/miniconda3/bin:$PATH"
	fi
fi
unset __conda_setup

conda activate tinytom


# Run your script
cd ~/procedural-evals-tom/code/src/training
torchrun --standalone --nproc_per_node=8 train.py --config ../../configs/training-neo-300.json

