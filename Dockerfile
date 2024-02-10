FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    gcc-9 \
    g++-9 \
    pip \
    python3-dev \
    python3 \
    wget \
    intel-mkl-full \
    libgoogle-glog-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libfreeimage3 \
    libfreeimage-dev \
    xorg-dev \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY External External
RUN cd External && \
    wget -nv https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip -O  libtorch.zip && \
    unzip libtorch.zip -d .

RUN wget -nv https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz && \
    tar xzf cmake-3.28.3-linux-x86_64.tar.gz

ENV CC=gcc-9
ENV CXX=g++-9
ENV CUDAHOSTCXX=g++-9

COPY cmake cmake
COPY configs configs
COPY loss loss
COPY scenes scenes
COPY shader shader
COPY src src
COPY .clang-format .
COPY CMakeLists.txt .

RUN mkdir build && \
    cd build && \
    ../cmake-3.28.3-linux-x86_64/bin/cmake -DCMAKE_PREFIX_PATH="./External/libtorch/;" .. && \
    make -j10
