#!/bin/bash -l
#SBATCH --job-name=gpu_job
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=course_gpu
#SBATCH --qos=course
#SBATCH --account=2026-spring-ece-381-th36-ucid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=59:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M

module load Miniforge3/24.11.3-0
module load CUDA/12.6.0

cd /course/2026/spring/ece/381/th36/$LOGNAME

conda activate fine-tune
python prepare_dataset.py
python finetune.py
