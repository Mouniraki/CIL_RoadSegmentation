#!/bin/bash
#SBATCH --account=3dv
#SBATCH --nodes=1                  # 24 cores
#SBATCH --gpus=1
###SBATCH --gres=gpumem:24g
#SBATCH --time 00:30:00        ### adapt to our needs
#SBATCH --mem-per-cpu=12000
###SBATCH -J analysis1
#SBATCH -o installation%j.out
#SBATCH -e installation%j.err
###SBATCH --mail-type=END,FAIL

. /etc/profile.d/modules.sh

echo "working"

python3 -m venv env-cil

source /home/tducrey/env-cil/bin/activate

cd /home/tducrey/CIL_RoadSegementation

#### put python commands here

pip install opencv-python
pip install matplotlib
pip install scikit-learn
pip install torch
pip install tqdm
pip install Augmentor




echo "finished"
