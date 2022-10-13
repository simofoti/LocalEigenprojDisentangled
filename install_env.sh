#!/bin/bash

virtualenv -p python3 ./id-generator-env
source ./id-generator-env/bin/activate

export PYTHON_V=38
export CUDA=cu113
export TORCH=1.12.1
export TORCHVISION=0.13.1
export TORCH_GEOM_VERSION=1.12.0

export TORCH_3D_VERSION="${TORCH_GEOM_VERSION//.}"

pip install cmake
pip install trimesh pyrender tqdm matplotlib rtree openmesh tb-nightly av seaborn joypy menpo

# install pytorch, pytorch geometric, and pytorch3d. make sure the correct variables were exported.
pip3 install torch==${TORCH} torchvision==${TORCHVISION} --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_GEOM_VERSION}+${CUDA}.html
pip install pytorch3d==0.6.0  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PYTHON_V}_${CUDA}_pyt${TORCH_3D_VERSION}/download.html

pip install geomloss pykeops

# for demo in notebook
pip install jupyterlab pythreejs
jupyter labextension install jupyter-threejs