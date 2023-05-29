FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/MADRID apt-get install -y tzdata

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

RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/NVlabs/nvdiffrast/

# Needs nvidia runtime, if you have "No CUDA runtime is found" error: https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime, first answer
RUN pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

RUN pip3 install git+https://github.com/openai/CLIP.git
RUN bash scripts/install_ext.sh


# Set the default command to python3
#CMD ["python3", "app.py"]

