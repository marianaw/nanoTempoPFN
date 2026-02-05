# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /workspace

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    tzdata \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -s /usr/bin/python3.12 /usr/bin/python

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip
RUN python3.12 -m pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt /workspace/

# Install Python dependencies
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Clone xlstm-jax library (shallow clone for speed)
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/NX-AI/xlstm-jax.git /tmp/xlstm-jax

# Install xlstm-jax library
RUN cd /tmp/xlstm-jax && python3.12 -m pip install --no-cache-dir -e .

# Clean up git files to save space
RUN rm -rf /tmp/xlstm-jax/.git

# Copy project files
COPY . /workspace/

# Create directories for persistent storage
RUN mkdir -p /workspace/checkpoints /workspace/persistent

# Set Python path
# Set PYTHONPATH for your code
ENV PYTHONPATH=/workspace
# ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden in RunPod)
CMD ["python3", "train.py", "--save-path", "/workspace/checkpoints/pretrained_model.pkl"]
