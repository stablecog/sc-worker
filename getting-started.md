## Install Pyenv

```
sudo apt update
python -m ensurepip --upgrade
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
        ca-certificates
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH" && eval "$(pyenv init --path)" && echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
```

## Install Python 3.10

```
pyenv install 3.10
exec "$SHELL"
```

## Initial Setup

```
virtualenv -p python3.10 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running after initial setup

```
source venv/bin/activate
```
