# Use an official Ubuntu base image
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install prerequisites
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update

# Install Python 3.10 and pip
RUN apt-get install -y python3.10 python3.10-distutils python3.10-venv python3-pip

# Set Python 3.10 as the default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Verify installation
RUN python --version && pip --version

# Create venv
ENV VENV_DIR=/venv
RUN python -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

# Install L4CasADi
# COPY . /l4acados
# WORKDIR /l4acados/external/l4casadi

RUN git clone https://github.com/IntelligentControlSystems/l4acados.git /opt/l4acados || exit 1
WORKDIR /opt/l4acados
RUN git submodule update --init --recursive || exit 1
# COPY ./external/l4casadi/requirements_build.txt /opt/l4casadi/requirements_build.txt

WORKDIR /opt/l4acados/external/l4casadi
RUN pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu || exit 1
RUN pip install numpy || exit 1
RUN pip install -r requirements_build.txt || exit 1
RUN pip install --no-build-isolation . || exit 1
# RUN pip install --no-build-isolation git+https://github.com/Tim-Salzmann/l4casadi.git
# RUN pip install . --no-build-isolation || exit 1

# # # Install L4acados
# WORKDIR /l4acados
WORKDIR /opt
ENV CMAKE_NAME=cmake-3.31.9-linux-x86_64
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.31.9/$CMAKE_NAME.tar.gz"
RUN tar -xzf "$CMAKE_NAME.tar.gz"

WORKDIR /opt/l4acados/external/acados
# RUN git clone https://github.com/acados/acados.git /opt/acados || exit 1
# RUN git checkout d108230ee
# RUN git submodule update --init --recursive || exit 1
RUN mkdir -p build && cd build && /opt/$CMAKE_NAME/bin/cmake -DACADOS_PYTHON=ON .. && make install -j4 || exit 1
RUN cd /opt/l4acados/external/acados/bin && wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -O t_renderer && chmod +x t_renderer || exit 1

WORKDIR /opt/l4acados
RUN pip install -e ./external/acados/interfaces/acados_template || exit 1
RUN pip install -e .[test] --no-build-isolation || exit 1
RUN pip install -e .[gpytorch-exo] || exit 1
