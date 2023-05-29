FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/MADRID apt-get install -y tzdata
ARG STABLE_DREAMFUSION_COMMIT_HASH="20e4214"
ARG NVDIFFRAST_COMMIT_HASH="335cfa6"
ARG TINY_CUDA_NN_COMMIT_HASH="a77dc53"
ARG CLIP_COMMIT_HASH="a9b1bf5"

# Install some basic utilities
RUN apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3 \
    python3-pip \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app

RUN cd /app && git clone https://github.com/thegenerativegeneration/stable-dreamfusion.git

WORKDIR /app/stable-dreamfusion

RUN git checkout $STABLE_DREAMFUSION_COMMIT_HASH

RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/NVlabs/nvdiffrast.git@$NVDIFFRAST_COMMIT_HASH

# Needs nvidia runtime, if you have "No CUDA runtime is found" error: https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime, first answer
RUN pip3 install "git+https://github.com/NVlabs/tiny-cuda-nn.git@$TINY_CUDA_NN_COMMIT_HASH#subdirectory=bindings/torch"

RUN pip3 install git+https://github.com/openai/CLIP.git@$CLIP_COMMIT_HASH
RUN bash scripts/install_ext.sh


# Set the default command to python3
#CMD ["python3", "app.py"]

