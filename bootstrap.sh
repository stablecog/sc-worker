#!/bin/bash

BOOTSTRAP_FILE=$HOME/sc-bootstrapped
PYTHON_TARGET_VERSION=3.10

# This script is used to bootstrap the environment for the StableCog worker

# See if sc-bootstrapped exists in the home directory
if [ -f $BOOTSTRAP_FILE ]; then
    echo "StableCog already bootstrapped"
    exit 0
fi

echo "🤖 Installing system dependencies..."
sudo env DEBIAN_FRONTEND=noninteractive  apt-get update
if [ $? -ne 0 ]; then
    echo "Failed to run apt-get"
    exit 1
fi
sudo env DEBIAN_FRONTEND=noninteractive  apt-get upgrade -y
if [ $? -ne 0 ]; then
    echo "Failed to run apt-get upgrade"
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
        python3-virtualenv


if [ $? -ne 0 ]; then
    echo "❌ Failed to install system dependencies"
    exit 1
fi

echo "🐍 Installing Python $PYTHON_TARGET_VERSION..."
curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash 
if [ $? -ne 0 ]; then
    echo "❌ Failed to install pyenv"
    exit 1
fi

# Setup pyenv if pyenv doesnt exist
if [ ! -d $HOME/.pyenv ]; then
    echo "🐍 Setting up pyenv..."
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.bashrc
    echo 'eval "$(pyenv init -)"' >> $HOME/.bashrc
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
if ! command -v $PYTHON_TARGET_VERSION &> /dev/null
then
    pyenv install-latest "$PYTHON_TARGET_VERSION" && pyenv global $(pyenv install-latest --print "$PYTHON_TARGET_VERSION") && pip install "wheel<1"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python $PYTHON_TARGET_VERSION"
        exit 1
    fi
fi

echo "📦 Installing StableCog worker dependencies..."
virtualenv -p $PYTHON_TARGET_VERSION venv && source venv/bin/activate && pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install StableCog worker dependencies"
    exit 1
fi

# Create bootstrap file
touch $BOOTSTRAP_FILE

echo "🎉 StableCog bootstrapped!"
exit 0