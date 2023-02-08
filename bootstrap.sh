#!/bin/bash

BOOTSTRAP_FILE=~/sc-bootstrapped
PYTHON_TARGET_VERSION=3.10

# This script is used to bootstrap the environment for the StableCog worker

# See if sc-bootstrapped exists in the home directory
if [ -f $BOOTSTRAP_FILE ]; then
    echo "StableCog already bootstrapped"
    exit 0
fi

echo "🤖 Installing system dependencies..."
sudo apt-get update
if [ $? -ne 0 ]; then
    echo "Failed to run apt-get"
    exit 1
fi
sudo apt-get upgrade -y
if [ $? -ne 0 ]; then
    echo "Failed to run apt-get upgrade"
    exit 1
fi

sudo apt-get install -y \
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        git \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0

if [ $? -ne 0 ]; then
    echo "❌ Failed to install system dependencies"
    exit 1
fi

echo "🐍 Installing Python $PYTHON_TARGET_VERSION..."
curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
        git clone https://github.com/momo-lab/pyenv-install-latest.git "$(pyenv root)"/plugins/pyenv-install-latest && \
        pyenv install-latest "$PYTHON_TARGET_VERSION" && \
        pyenv global $(pyenv install-latest --print "$PYTHON_TARGET_VERSION") && \
        pip install "wheel<1"

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python $PYTHON_TARGET_VERSION"
    exit 1
fi

# Create bootstrap file
touch $BOOTSTRAP_FILE

echo "🎉 StableCog bootstrapped!"
exit 0