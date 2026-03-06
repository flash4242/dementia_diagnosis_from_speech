FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Install necessary system packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    tmux \
    vim \
    htop \
    openssh-server \
    zip \
    unzip \
    build-essential \
    graphviz \
    tree \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set the environment variable for Matplotlib to avoid cache issues
ENV MPLCONFIGDIR=/tmp

# Set the working directory (defaults to root's workspace)
WORKDIR /workspace
