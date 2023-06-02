#!/bin/bash

BOOTSTRAP_FILE=./sc-bootstrapped
PYTHON_TARGET_VERSION=3.10

# This script is used to bootstrap the environment for the Stablecog worker

# See if sc-bootstrapped exists in the home directory
if [ -f $BOOTSTRAP_FILE ]; then
    echo "Stablecog already bootstrapped"
    exit 0
fi

echo "🤖 Installing system dependencies..."
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
if [ $? -ne 0 ]; then
    echo "Failed to run apt-get"
    exit 1
fi

sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
    libglib2.0-0 \
    ffmpeg

if [ $? -ne 0 ]; then
    echo "❌ Failed to install system dependencies"
    exit 1
fi

echo "🐍 Installing Python $PYTHON_TARGET_VERSION..."
# Setup pyenv if pyenv doesnt exist
if [ ! -d $HOME/.pyenv ]; then
    echo "🐍 Setting up pyenv..."
    curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install pyenv"
        exit 1
    fi
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >>$HOME/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >>$HOME/.bashrc
    echo 'eval "$(pyenv init -)"' >>$HOME/.bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    ! command -v pyenv >/dev/null && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    git clone https://github.com/momo-lab/pyenv-install-latest.git "$(pyenv root)"/plugins/pyenv-install-latest
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install pyenv-install-latest"
        exit 1
    fi
fi

# install python_target_version if not in path
pyenv install-latest "$PYTHON_TARGET_VERSION" && pyenv global $(pyenv install-latest --print "$PYTHON_TARGET_VERSION") && pip install "wheel<1"
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python $PYTHON_TARGET_VERSION"
    exit 1
fi
python$PYTHON_TARGET_VERSION -m pip install virtualenv

if [ $? -ne 0 ]; then
    echo "❌ Failed to install virtualenv"
    exit 1
fi

echo "📦 Installing Stablecog worker dependencies..."
python$PYTHON_TARGET_VERSION -m virtualenv venv && source venv/bin/activate && pip install -r requirements-torch.txt && pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Stablecog worker dependencies"
    exit 1
fi

echo "Making start script executable"
sudo chmod +x ./run

# Create bootstrap file
touch $BOOTSTRAP_FILE

echo "🎉 Stablecog bootstrapped!"
echo "🔑 Don't forget to create a ".env" file"
echo "🎬 You can start the server by running './run.sh'"
echo "Please run 'source ~/.bashrc' or reload your terminal"
exit 0
