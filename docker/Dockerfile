############################################
# Dockerfile for Puppeteer                 #
# Nicklas Hansen, 2024 (c)                 #
# https://www.tdmpc2.com                   #
# ---------------------------------------- #
# Build instructions:                      #
# docker build . -t <user>/puppeteer:1.0.0 #
# docker push <user>/puppeteer:1.0.0       #
# ---------------------------------------- #
# Run:                                     #
# docker run -i \                          #
#   -v <path>/<to>/puppeteer:/puppeteer \  #
#   --gpus all \                           #
#   -t <user>/puppeteer:1.0.0 \            #
#   /bin/bash                              #
############################################

# base image
FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl \
    wget unzip htop tmux xvfb patchelf ca-certificates bash-completion libjpeg-dev libpng-dev \
    ffmpeg cmake swig libssl-dev libcurl4-openssl-dev libopenmpi-dev python3-dev zlib1g-dev \
    qtbase5-dev qtdeclarative5-dev libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 \
    libgl1-mesa-glx libosmesa6 libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda init && \
    conda clean -ya
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# conda environment
COPY environment.yaml /root
RUN conda update conda && \
    conda env update -n base -f /root/environment.yaml && \
    rm /root/environment.yaml && \
    conda clean -ya && \
    pip cache purge

# success!
RUN echo "Successfully built Puppeteer Docker image!"
