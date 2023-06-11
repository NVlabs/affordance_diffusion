set -e 
set -x

# conda create -n affordance_diffusion python=3.8
# conda activate affordance_diffusion

# pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y


# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y 
pip install ninja
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.0"


pip install -r requirements.txt



# Install Frankmocap
# rm -r third_party/frankmocap
# mkdir -p externals
# # my modification on relative path
# git clone https://github.com/judyye/frankmocap.git externals/frankmocap
cd thrid_party/frankmocap
bash scripts/install_frankmocap.sh
cd ../..

# # detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

