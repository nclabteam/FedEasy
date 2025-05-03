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
2. After cloning, navigate to the cloned directory and open the terminal.

3. Ensure `pip` is installed. If not, install it using:
```bash
 sudo apt install python3-pip
```

4. We will use Miniconda to create a virtual environment. Download Miniconda with: (*If you already have Conda installed, skip steps 4 and 5.*)

```bash
 curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
```
5. Then install Miniconda using the command below:
```bash
  bash Miniconda3-latest-Linux-x86_64.sh
```
6. Create a new virtual environment using Conda:
```bash
 conda env create -f environment.yaml
```
This will create a virtual environment named `venv-fedeasy` based on the `environment.yaml` file.

7. Activate the virtual environment:
```bash
 conda deactivate
 conda activate venv-fedeasy
```
8. You can change the configuration as per your needs in the `config.yaml` file.

9. To scale clients to a few hundred, we can run Flower in simulation mode on a single machine like this:
```bash
python main.py
    or
python main.py --config /path/to/config.yaml
```
  This script reads the configuration from the `config.yaml` file and starts the simulation.

  The outputs will be saved in the `out` directory.