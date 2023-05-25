FROM nvcr.io/nvidia/pytorch:21.05-py3

ENV DEBIAN_FRONTEND=noninteractive 

# library
RUN apt-get update && apt-get install -y \
    libgl1 \
    xvfb \
    htop \
    libx11-6  \
    ffmpeg 


# pytorch3d
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y 
RUN conda install -c bottler nvidiacub -y 


# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

WORKDIR /usr/src/app

RUN pip install -U pip
RUN pip --no-cache-dir install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
# used by frankmocap. You can comment it out if 3D prediction is not needed.
RUN pip --no-cache-dir install "git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13"


COPY requirement.txt requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

WORKDIR /workspace