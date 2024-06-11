#!/usr/bin/env bash

env=$1


echo "Creating virtual env.: $env"
rm -r ~/venvs/$env
python3.10 -m venv ~/venvs/$env
source ~/venvs/$env/bin/activate

echo "Installing..."
pip install -r reqs_sr.micro.txt

pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install pretrainedmodels zipp timm efficientnet-pytorch  kornia
pip install seaborn

pip install einops
pip install omegaconf
pip install fairscale

cd dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install

deactivate

echo "Done creating and installing virt.env: $env."