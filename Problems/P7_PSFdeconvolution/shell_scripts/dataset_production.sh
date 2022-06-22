#!/bin/bash
#SBATCH --account=def-hezaveh
#SBATCH --array=0-39
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00

source $HOME/venvs/astromatic/bin/activate

python $ASTROMATIC_PATH/Problems/P7_PSFdeconvolution/generate_training_data.py\
  --size=50000\
  --rpf=1000\
  --dataset=deconv_50k_dataset\
  --npix=256\
  --psf_npix=75