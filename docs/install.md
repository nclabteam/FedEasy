# Installation

> **Note**: This code has been written and tested on Ubuntu and should work seamlessly on any Linux-based distribution. Windows users may need to adjust some steps.

## 📋 Prerequisites

- Git
- Python 3.x
- pip (Python package installer)
- Miniconda (recommended for environment management)

## 🚀 Getting Started

Follow these steps to set up and use this repository:

1. Clone the Repository

```bash
    git clone https://github.com/nclabteam/FedEasy.git
```
2. After clone cd into cloned directory and open terminal.

3. Ensure pip is installed if not install using
```bash
    sudo apt install python3-pip
```

4. We will be using miniconda to create a virtual environment, download miniconda as
If You already have conda installed skip step 4 and 5.

```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
```
5. Then install miniconda using below command
```bash
    bash Miniconda3-latest-Linux-x86_64.sh
```
6. Create a new virtual environment using conda
```bash
    conda env create -f environment.yaml
```
It will create a virtual env named `venv-fedeasy` based on `environment.yaml` file

7. For using virtual environment we need to activate the environment first.
```bash
    conda deactivate
    conda activate venv-fedeasy
```
8. We can change the confugration as per our need in config.yaml file

9. We can run scale clients to few hundred we can run flower in simulation mode on single machine like:
```bash
    python main.py
            or
    python main.py --config /path/to/config.yaml
```
This script will read the confugration from `config.yaml` file and starts the simulation.

The outputs will be saved in `out` directory.


