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

source /cluster/courses/3dv/data/team-4/paper_imported_code/DROID-SLAM/env-droid/bin/activate

cd /cluster/courses/3dv/data/team-4/paper_imported_code/DROID-SLAM

#### put python commands here

# ./tools/validate_tartanair.sh --plot_curve

TARTANAIR_PATH=datasets/TartanAir
python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --reconstruction_path P001_reconstruction --weights=droid.pth --disable_vis --buffer=650 $@


echo "finished"
