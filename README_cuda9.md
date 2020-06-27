# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound for Multiple Object Tracking

## Experts

* [DAN](https://arxiv.org/abs/1810.11780)[<https://github.com/shijieS/SST>]
* [DeepMOT](https://arxiv.org/abs/1906.06618)[<https://gitlab.inria.fr/yixu/deepmot>]
* [DeepSORT](https://arxiv.org/abs/1812.00442)[<https://github.com/nwojke/deep_sort>]
* [MOTDT](https://arxiv.org/abs/1809.04427)[<https://github.com/longcw/MOTDT>]
* [SORT](https://arxiv.org/abs/1602.00763)[<https://github.com/abewley/sort>]
* [Tracktor](https://arxiv.org/abs/1903.05625)[<https://github.com/phil-bergmann/tracking_wo_bnw>]

## Datasets

* [MOT16, MOT17](https://arxiv.org/abs/1603.00831)[<https://motchallenge.net>]
* [MOT20](https://arxiv.org/abs/2003.09003)[<https://motchallenge.net>]

## Frameworks

* py-motmetrics[<https://github.com/cheind/py-motmetrics>] for evaluating trackers.

## Setup

1. Install [Cuda 9.2](https://developer.nvidia.com/cuda-92-download-archive)

    ```sh
    Do you accept the previously read EULA?
    accept/decline/quit: accept

    You are attempting to install on an unsupported configuration. Do you wish to continue?
    (y)es/(n)o [ default is no ]: y

    Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 xxxxx?
    (y)es/(n)o/(q)uit: n

    Install the CUDA 9.2 Toolkit?
    (y)es/(n)o/(q)uit: y

    Enter Toolkit Location
    [ default is /usr/local/cuda-9.2 ]:

    Do you want to install a symbolic link at /usr/local/cuda?
    (y)es/(n)o/(q)uit: n

    Install the CUDA 9.2 Samples?
    (y)es/(n)o/(q)uit: n
    ```

    If Compiled error occurs, try to install gcc++-6

    ```sh
    sudo apt install gcc-6 g++-6
    sudo unlink /usr/local/bin/gcc
    sudo unlink /usr/local/bin/g++
    sudo ln -s /usr/bin/gcc-6 /usr/local/bin/gcc
    sudo ln -s /usr/bin/g++-6 /usr/local/bin/g++
    ```

2. Make sure that Cuda 9.2 is used

    ```sh
    $ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2018 NVIDIA Corporation
    Built on Tue_Jun_12_23:07:04_CDT_2018
    Cuda compilation tools, release 9.2, V9.2.148
    ```

    If the different version of nvcc is displayed, try to install a symbolic link

    ```sh
    sudo unlink /usr/local/cuda
    sudo ln -s /usr/local/cuda-9.2 /usr/local/cuda
    ```

    And replace `/usr/local/cuda-[SOME VERSION]` with `/usr/local/cuda` in ~/.bashrc

    ```sh
    gedit ~/.bashrc
    source ~/.bashrc
    ```

## Requirements

1. Clone the repository of experts

    ```sh
    mkdir external
    cd external
    git clone https://github.com/shijieS/SST.git
    git clone https://gitlab.inria.fr/yixu/deepmot.git
    git clone https://github.com/nwojke/deep_sort.git
    git clone https://github.com/longcw/MOTDT.git
    git clone https://github.com/abewley/sort.git
    git clone --recurse-submodules --branch iccv_19 https://github.com/songheony/tracking_wo_bnw.git tracking_wo_bnw_cuda9
    ```

2. Install necessary libraries with Anaconda 3

    ```sh
    conda create -n [ENV_NAME] python=3.6
    conda activate [ENV_NAME]

    # For our framework
    conda install -y black flake8 pandas

    # For DeepMOT and Tracktor, it is needed to install pytorch <= 0.4.1
    # For FPN which is used in Tracktor, some error occurs with pytorch==0.4.1
    conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch

    # For DeepSORT, it is needed to install tensorflow <= 2.0
    conda install -y tensorflow-gpu==1.14.0 cudatoolkit==9.2

    # For experts
    conda install -y pillow pyyaml==3.13 cython matplotlib scikit-learn==0.22.1 scikit-image tqdm numba
    conda install -y -c conda-forge opencv easydict filterpy
    pip install motmetrics lapsolver
    ```

3. Build FPN and FRCNN from Tracktor

    ```sh
    cd external/tracking_wo_bnw_cuda9
    conda activate [ENV_NAME]
    bash src/fpn/fpn/make.sh
    bash src/frcnn/frcnn/make.sh
    ```

## How to run

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
