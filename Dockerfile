FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Replace with owner UID/GID
ARG UID=1000 \
    GID=1000 \
    REQUIREMENTS=./requirements.txt \
    SERVICE_NAME=app

# Add new user
ENV USERNAME=user
ENV HOME=/home/$USERNAME
ENV DEBIAN_FRONTEND=noninteractive

# Create the user and set up environment
RUN useradd -m -u $UID -s /bin/bash $USERNAME && \
    mkdir /etc/sudoers.d && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    groupmod --gid $GID $USERNAME

# Install essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3.10 \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
ENV PATH="$HOME/.local/bin:$PATH"
ENV PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Install PyTorch
RUN pip install --no-cache-dir --timeout 60 \
    -f https://download.pytorch.org/whl/torch/ torch==2.4.1

# Copy and install Python dependencies
COPY $REQUIREMENTS $HOME/$SERVICE_NAME/requirements.txt
WORKDIR $HOME/$SERVICE_NAME
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --user --no-cache-dir --timeout 60 -r requirements.txt && \
    rm requirements.txt

# Copy environment file and application code
COPY .env .
COPY . .

# Command to run the application
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]