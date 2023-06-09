## Installation guide
1. Recursively clone this github repo `git clone --recursive https://github.com/NVlabs/affordance_diffusion.git`
1. Download [pre-trained models]() and unzip them to `output/`
1. (Optionally) Install third party pakcages  if you want to run 3D hand pose prediction and render it.
- Follow [instructions](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md) from Frankmocap and put it under `third_party/frankmocap` 
- Download `MANO_RIGHT.pkl` from the [official website](https://mano.is.tue.mpg.de/) and put it under `thrid_party/mano/MANO_RIGHT.pkl`


## Environment specification
- pytorch=1.9
- pytorch-lightning=1.6.5
- pytorch3d=0.7.0
- detectron=0.6.0 (used by frankmocap)
