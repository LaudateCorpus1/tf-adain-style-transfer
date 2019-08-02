FROM nvidia/cuda:9.0-base-ubuntu16.04

# install libraries for python3, CUDA, cuDNN, plus utilities git, vim, wget, (un)zip, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        curl \
        wget \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libcudnn7-dev=7.1.4.18-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        libopencv-dev \
        pkg-config \
        libzmq3-dev \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-tk \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        cmake \
        libreadline-dev \
        graphicsmagick \
        libgtk-3-dev \
        libboost-all-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a


# install the most up to date version of python3 package manager
RUN pip3 install --upgrade pip

# core python3 modules
RUN pip3 --no-cache-dir install \
    backports.weakref==1.0rc1 \
    bleach==1.5.0 \
    cycler==0.10.0 \
    decorator==4.1.2 \
    h5py==2.7.0 \
    html5lib==0.9999999 \
    Markdown==2.6.8 \
    matplotlib==2.0.2 \
    networkx==1.11 \
    numpy==1.13.3 \
    olefile==0.44 \
    Pillow==4.2.1 \
    protobuf==3.5.1 \
    pyparsing==2.2.0 \
    python-dateutil==2.6.1 \
    pytz==2017.2 \
    PyWavelets==0.5.2 \
    scikit-image==0.13.0 \
    scipy==0.19.1 \
    six==1.10.0 \
    tensorflow-gpu==1.8 \
    Werkzeug==0.12.2

COPY . /tf-adain
WORKDIR /tf-adain

RUN mkdir my_output

RUN rm -rf models && mkdir models && cd models && wget https://s3-us-west-2.amazonaws.com/deepai-opensource-codebases-models/tensorflow-fast-style-transfer/decoder_weights.h5

RUN cd models && wget https://s3-us-west-2.amazonaws.com/deepai-opensource-codebases-models/tensorflow-fast-style-transfer/vgg19_weights_normalized.h5

RUN pip3 install ai-integration==1.0.11

ENTRYPOINT ["python3", "entrypoint.py"]

