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
module load cuda/12.1
export CUB_HOME=/cluster/courses/3dv/data/team-4/tjark_environment_test/DPT-SLAM/thirdparty/DOT/dot/utils/torch3d/cub-2.1.0
echo $CUB_HOME
export CXXFLAGS="-std=c++17"


echo "working"

source /cluster/courses/3dv/data/team-4/tjark_environment_test/DPT-SLAM/env_dpt/bin/activate

cd /cluster/courses/3dv/data/team-4/tjark_environment_test/DPT-SLAM

#### put python commands here

pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install tensorboard opencv-python scipy tqdm suitesparse-graphblas matplotlib PyYAML gdown ninja
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install evo --upgrade --no-binary evo

python setup.py install

pip install einops einshape timm lmdb av mediapy

cd thirdparty/DOT/dot/utils/torch3d/ && pip install . && cd ../../..

# ./tools/validate_tartanair.sh

echo "finished"
