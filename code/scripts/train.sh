#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres=gpu:5
#SBATCH --mem=480G
#SBATCH --cpus-per-task=96
#SBATCH --time=7-0
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load any necessary modules or environment variables here
# For example:



__conda_setup="$('/scr/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
    else
	if [ -f "/scr/miniconda3/etc/profile.d/conda.sh" ]; then
		. "/scr/miniconda3/etc/profile.d/conda.sh"
	else
		export PATH="/scr/miniconda3/bin:$PATH"
	fi
fi
unset __conda_setup

conda activate tinytom

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Run your script
cd ~/procedural-evals-tom/code/src/training
torchrun --standalone --nproc_per_node=4 train.py --config ../../configs/training-neo-125.json

