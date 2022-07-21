#!/bin/bash

#source $HOME/venvs/astromatic/bin/activate

python $ASTROMATIC_PATH/Problems/P7_PSFdeconvolution/desi_downloader_script.py\
  --size=200000\
  --dataset_name=cuts_256_200k\
  --npix=256