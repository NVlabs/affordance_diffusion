## Installation guide
1. Recursively clone this github repo `git clone --recursive https://github.com/NVlabs/affordance_diffusion.git`
1. Download [pre-trained models](https://drive.google.com/file/d/1Ny8wtQJL1TFa3qYsxUnl4BPynpJEMaMB/view?usp=sharing) and unzip them to `${environment.output}` specified in [configs/environment/grogu.yaml](configs/environment/grogu.yaml). Let us assume it is `output/`
    ```
    mkdir -p output/
    cd output/
    gdown 1Ny8wtQJL1TFa3qYsxUnl4BPynpJEMaMB
    tar xfz release.tar.gz
    cd - 
    ```
1. Install third party packages  if you want to extract 3D hand pose from the generated hands and overlay its rendering.
    - Install frankmocap and pytorch3d, etc. 
        ```
        bash scripts/install_thrid_party.sh
        ``` 
    - Download `MANO_RIGHT.pkl` from the [official website](https://mano.is.tue.mpg.de/) and put it under `thrid_party/mano/MANO_RIGHT.pkl`


## Environment specification
- pytorch=1.10
- pytorch-lightning=1.6.5
- pytorch3d=0.7.0
- detectron=0.6.0 (used by frankmocap)
