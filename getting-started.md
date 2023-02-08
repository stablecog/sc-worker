## Install Pyenv

```
sudo apt update
curl https://pyenv.run | bash
```

## Install Python 3.10

```
pyenv install 3.10
```

## Setup

```
virtualenv -p python3.10 venv
source venv/bin/activate
pip install -r requirements.txt
```
