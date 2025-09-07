FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    locales \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Jazzy
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-tf-transformations \
    ros-jazzy-vision-msgs \
    python3-rosdep \
    python3-colcon-common-extensions \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# Install opengl libraries
RUN apt-get update && apt-get install -y \
    libglvnd0 libgl1 libglx0 libegl1 \
    libx11-6 libxext6 libxrandr2 libxrender1 \
    libxkbcommon0 libx11-xcb1 libxcb1 libxcb-dri3-0 libxcb-present0 \
    mesa-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

# Install megapose dependencies
WORKDIR /megapose_ros_ws

COPY src/megapose/megapose/deps/bop_toolkit_challenge/ /opt/bop_toolkit_challenge/

RUN pip install --no-cache-dir /opt/bop_toolkit_challenge --break-system-packages

RUN apt-get update && apt-get remove -y python3-kiwisolver python3-contourpy

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt --break-system-packages

# Clean up temporary files
RUN rm -rf /tmp/bop_toolkit_challenge /opt/requirements.txt

CMD ["/bin/bash"]