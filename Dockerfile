FROM nvidia/cuda:9.0-base-ubuntu16.04

# install libraries for python3, CUDA, cuDNN, plus utilities git, vim, wget, (un)zip, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential=12.1ubuntu2 \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        curl=7.47.0-1ubuntu2.12 \
        wget=1.17.1-1ubuntu1.5 \
        vim=2:7.4.1689-3ubuntu1.2 \
        git=1:2.7.4-0ubuntu1.6 \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libcudnn7-dev=7.1.4.18-1+cuda9.0 \
        libfreetype6-dev=2.6.1-0.1ubuntu2.3 \
        libhdf5-serial-dev=1.8.16+docs-4ubuntu1.1 \
        libpng12-dev=1.2.54-1ubuntu1.1 \
        libzmq3-dev=4.1.4-7 \
        libopencv-dev=2.4.9.1+dfsg-1.5ubuntu1.1 \
        pkg-config=0.29.1-0ubuntu1 \
        libzmq3-dev=4.1.4-7 \
        python3-pip=8.1.1-2ubuntu0.4 \
        python3-dev=3.5.1-3 \
        python3-setuptools=20.7.0-1 \
        python3-tk=3.5.1-1 \
        rsync=3.1.1-3ubuntu1.2 \
        software-properties-common=0.96.20.8 \
        unzip=6.0-20ubuntu1 \
        zip=3.0-11 \
        zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.1 \
        cmake=3.5.1-1ubuntu3 \
        libreadline-dev=6.3-8ubuntu2 \
        graphicsmagick=1.3.23-1ubuntu0.1 \
        libgtk-3-dev=3.18.9-1ubuntu3.3 \
        libboost-all-dev=1.58.0.1ubuntu1 \
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

RUN pip3 install ai-integration==1.0.6

ENTRYPOINT ["python3", "entrypoint.py"]

