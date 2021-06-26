
FROM pytorch/pytorch


SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        xvfb \
        ffmpeg \
        xorg-dev \
        libsdl2-dev \
        swig \
        cmake \
        python-opengl \
        tmux \
        wget \
        unrar \
        unzip 
        
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
                 gym \
                 box2d-py \
                 matplotlib \
                 seaborn \
                 pandas \
                 notebook \
                 scikit-image \
                 atari_py

RUN wget http://www.atarimania.com/roms/Roms.rar &&  unrar e Roms.rar && unzip ROMS.zip && python3 -m atari_py.import_roms ROMS

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
