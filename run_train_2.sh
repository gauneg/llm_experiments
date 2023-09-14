#!/bin/sh
#SBATCH --time=42:00:00
#SBATCH --nodes=1
#SBATCH -A ngeng206c
#SBATCH -p GpuQ

module load cuda/11.2
module load conda/2

source activate torch_experiments
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ichec/packages/conda/2/pkgs/pytorch-1.3.0-py3.7_cuda10.1.243_cudnn7.6.3_0/lib

cd /ichec/work/ngeng206c/llm_experiments

/ichec/home/users/gauneg/.conda/envs/torch_experiments/bin/python \
/ichec/work/ngeng206c/llm_experiments/train_with_gen_argkey2.py \
--train_key relabelled_lap_788 --cuda 0 & \
/ichec/home/users/gauneg/.conda/envs/torch_experiments/bin/python \
/ichec/work/ngeng206c/llm_experiments/train_with_gen_argkey2.py \
--train_key relabelled_res_904 --cuda 1 
#/ichec/home/users/gauneg/.conda/envs/torch_experiments/bin/python \
#/ichec/home/users/gauneg/llm_experiments/evaluations_extracted_terms.py \
#--model_path /ichec/work/ngeng206c/models/mean_all_sigma_lap_res2 --cuda_device 1
