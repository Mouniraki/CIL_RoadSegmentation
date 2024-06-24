#!/bin/bash
#SBATCH --account=3dv
#SBATCH --nodes 1                  # 24 cores
#SBATCH --gpus 1
###SBATCH --gres=gpumem:24g
#SBATCH --time 00:30:00        ### adapt to our needs
#SBATCH --mem-per-cpu=12000
###SBATCH -J analysis1
#SBATCH -o droid%j.out
#SBATCH -e droid%j.err
###SBATCH --mail-type=END,FAIL

. /etc/profile.d/modules.sh
module load cuda/12.1

echo "working"

source /home/tducrey/env-cil/bin/activate

cd /home/tducrey/CIL_RoadSegementation

#### put python commands here

# ./tools/validate_tartanair.sh --plot_curve

python main.py


echo "finished"
