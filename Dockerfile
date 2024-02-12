FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as build
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
    wget -nv https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip -O libtorch.zip && \
    unzip -qq libtorch.zip -d . && \
    rm libtorch.zip

RUN wget -nv https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz -O cmake-dist.tar.gz && \
    tar xzf cmake-dist.tar.gz && \
    mv cmake-3.28.3-linux-x86_64 cmake-dist && \
    rm cmake-dist.tar.gz

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
    ../cmake-dist/bin/cmake -DCMAKE_PREFIX_PATH="./External/libtorch/;" .. && \
    make -j6

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-mkl-full \
    libx11-dev \
    xserver-xorg-dev \
    xorg-dev \
    libprotobuf23 \
    libfreeimage3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/build/bin/ /app
COPY --from=build /app/configs/ /app/configs
COPY --from=build /app/shader/ /app/shader
COPY --from=build /app/loss/ /app/loss
COPY --from=build /app/External/saiga/shader/ /app/External/saiga/shader
COPY --from=build /app/build/External/saiga/src/saiga/core/libsaiga_core.so /app
COPY --from=build /app/build/External/saiga/src/saiga/opengl/libsaiga_opengl.so /app
COPY --from=build /app/build/External/saiga/src/saiga/cuda/libsaiga_cuda.so /app
COPY --from=build /app/build/External/saiga/submodules/assimp/bin/ /app
COPY --from=build /app/build/External/saiga/submodules/glfw/src/libglfw.so /app
COPY --from=build /app/build/External/saiga/submodules/glfw/src/libglfw.so.3 /app
COPY --from=build /app/build/External/saiga/submodules/glfw/src/libglfw.so.3.4 /app
COPY --from=build /app/build/External/saiga/submodules/glog/libglog.pc /app
COPY --from=build /app/build/External/saiga/submodules/glog/libglog.so /app
COPY --from=build /app/build/External/saiga/submodules/glog/libglog.so.1 /app
COPY --from=build /app/build/External/saiga/submodules/glog/libglog.so.0.6.0 /app
COPY --from=build /app/External/libtorch/lib/ /app
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/app/
