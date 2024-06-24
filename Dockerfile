FROM ubuntu:22.04

# Python 3.9
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Europe/Zurich apt-get install -y software-properties-common cmake wget
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Europe/Zurich apt-get install -y git python3.9 python3-pip python3.9-distutils

# Our package
# WORKDIR /home
# TODO: switch to public repo
# TODO: use from GitLab
# RUN git clone https://docker:x_YtL3UWZmy1G9bXuZBd@gitlab.ethz.ch/ics-group/projects/amon-lahr/zero-order-gp-mpc-package.git
# WORKDIR /home/zero-order-gp-mpc-package
# RUN git submodule update --recursive --init

# # Acados (maybe not needed if pulled from zero-order)
# # git clone https://github.com/acados/acados.git
# # git submodule update --recursive --init
# WORKDIR /home/zero-order-gp-mpc-package/external/acados/build
# RUN cmake -DACADOS_PYTHON=ON .. && make install -j4
# ENV ACADOS_SOURCE_DIR=/home/zero-order-gp-mpc-package/external/acados
# ENV LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib

# # Python dependencies
# WORKDIR /home/zero-order-gp-mpc-package
# COPY requirements.txt .
# RUN python3.9 -m pip install -r requirements.txt

# # tera_renderer
# WORKDIR /home/zero-order-gp-mpc-package/external/acados/bin
# RUN wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -O t_renderer && chmod +x t_renderer
