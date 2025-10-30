# Use an official Ubuntu base image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install prerequisites
RUN apt-get update && apt-get install -y curl git wget

# Create venv
ENV VENV_DIR=/venv
RUN python -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

# Install L4CasADi
ENV L4CASADI_SOURCE_DIR=/opt/l4casadi
WORKDIR $L4CASADI_SOURCE_DIR
RUN git clone https://github.com/Tim-Salzmann/l4casadi.git $L4CASADI_SOURCE_DIR || exit 1
RUN git checkout 9fe2894533e05009bcbfa7706966745c5236fa4c
RUN pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu || exit 1
RUN pip install numpy || exit 1
RUN pip install -r requirements_build.txt || exit 1
RUN pip install --no-build-isolation . || exit 1

# Install acados
ENV ACADOS_SOURCE_DIR=/opt/acados
WORKDIR $ACADOS_SOURCE_DIR
RUN git clone https://github.com/acados/acados.git $ACADOS_SOURCE_DIR || exit 1
RUN git checkout 5847c74fe774137a5fb31c9791bf39aa980b8b05 || exit 1
RUN git submodule update --init --recursive || exit 1
RUN mkdir -p build && cd build && cmake -DACADOS_PYTHON=ON .. && make install -j4 || exit 1
RUN pip install -e $ACADOS_SOURCE_DIR/interfaces/acados_template || exit 1
# tera_renderer installation
RUN cd $ACADOS_SOURCE_DIR/bin && wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -O t_renderer && chmod +x t_renderer || exit 1
