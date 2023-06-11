## Installation guide
1. Recursively clone this github repo `git clone --recursive https://github.com/NVlabs/affordance_diffusion.git`
1. Download [pre-trained models](https://drive.google.com/file/d/1Ny8wtQJL1TFa3qYsxUnl4BPynpJEMaMB/view?usp=sharing) and unzip them to `output/`
```
mkdir -p output/
cd output/
gdown 1Ny8wtQJL1TFa3qYsxUnl4BPynpJEMaMB
tar xfz release.tar.gz
cd - 
```
1. (Optionally) Install third party pakcages  if you want to run 3D hand pose prediction and render it.
- Follow [instructions](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md) from Frankmocap and put it under `third_party/frankmocap` 
- Download `MANO_RIGHT.pkl` from the [official website](https://mano.is.tue.mpg.de/) and put it under `thrid_party/mano/MANO_RIGHT.pkl`


## Environment specification
- pytorch=1.10
- pytorch-lightning=1.6.5
- pytorch3d=0.7.0
- detectron=0.6.0 (used by frankmocap)
