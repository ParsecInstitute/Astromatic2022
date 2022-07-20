#!/bin/bash

#source $HOME/venvs/astromatic/bin/activate

python $ASTROMATIC_PATH/Problems/P7_PSFdeconvolution/desi_downloader_script.py\
  --size=25\
  --dataset_name=dummy_cuts\
  --npix=128