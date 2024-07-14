#!/bin/bash
#SBATCH --account=cil
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

# TODO: MODIFY THE USERNAME WITH YOUR OWN
$USERNAME="moraki"

echo "working"
python3 -m venv /home/$USERNAME/env-cil

source /home/$USERNAME/env-cil/bin/activate

cd /home/$USERNAME/CIL_RoadSegmentation/utils/env_setup

#### put python commands here

pip install -r requirements.txt

echo "finished"
