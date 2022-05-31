#!/bin/bash
#SBATCH --account=def-hezaveh
#SBATCH --array=0-19
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00

source $HOME/venvs/astromatic/bin/activate

python $ASTROMATIC_PATH/Problems/P2_lens_inference/lensing_pipeline.py\
  --dataset_name=tbd
  --data_type=lens
  --size=1\
  --rpf=100\
  --npix=128\
  --zl=0.5\
  --zs=1.5