# Use an official Python base with build tools
FROM python:3.11-slim

# Install system dependencies for PyBullet GUI and Python packages
RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    python3-tk \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    libxi-dev \
    libxmu-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your repository into the container
COPY . .

# Install PyFlyt and its dependencies
# (Assuming your repo has a setup.py or pyproject.toml)
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install "pybullet>=3.2.0" matplotlib

# Set environment variables for PyBullet GUI
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1

# Default command (you can override this in docker run)
CMD ["bash"]
