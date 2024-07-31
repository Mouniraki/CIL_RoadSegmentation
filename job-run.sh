#!/bin/bash
#SBATCH --account=cil
#SBATCH --nodes 1                  # 24 cores
#SBATCH --gpus 1
###SBATCH --gres=gpumem:24g
#SBATCH --time 00:30:00        ### adapt to our needs
#SBATCH --mem-per-cpu=12000
###SBATCH -J analysis1
#SBATCH -o cil_road%j.out
#SBATCH -e cil_road%j.err
###SBATCH --mail-type=END,FAIL

. /etc/profile.d/modules.sh

# TODO: MODIFY THE USERNAME WITH YOUR OWN
USERNAME=""

module load cuda/12.1

echo "working"

source /home/$USERNAME/env-cil/bin/activate

cd /home/$USERNAME/CIL_RoadSegmentation

#### put python commands here

<<<<<<< HEAD
python3 postproc_pipeline.py --postprocessing_type=deepnet --batch_size=4
=======
python main.py --n_folds=5 --use_lr_scheduler --loss=diceloss --batch_size=4 --n_epochs=100 --use_transforms --n_augmentation=10 --early_stopping_threshold=10
>>>>>>> 21a8649d21ec80493e2d922e7095ef1c1a5163cd

echo "finished"
