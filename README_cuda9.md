# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound for Multiple Object Tracking

## Experts

* [DeepMOT](https://arxiv.org/abs/1906.06618)[<https://gitlab.inria.fr/yixu/deepmot>]
* [DAN](https://arxiv.org/abs/1810.11780)[<https://github.com/shijieS/SST>]
* [IOU](https://ieeexplore.ieee.org/document/8078516)[<https://github.com/bochinski/iou-tracker>]
* [VIOU](https://ieeexplore.ieee.org/document/8639144)[<https://github.com/bochinski/iou-tracker>]
* [Tracktor](https://arxiv.org/abs/1903.05625)[<https://github.com/phil-bergmann/tracking_wo_bnw/tree/iccv_19>]
* [DeepSORT](https://arxiv.org/abs/1812.00442)[<https://github.com/nwojke/deep_sort>]
* [SORT](https://arxiv.org/abs/1602.00763)[<https://github.com/abewley/sort>]
* [MOTDT](https://arxiv.org/abs/1809.04427)[<https://github.com/longcw/MOTDT>]

## Datasets

* [MOT16, MOT17](https://arxiv.org/abs/1603.00831)[<https://motchallenge.net>]
* [MOT20](https://arxiv.org/abs/2003.09003)[<https://motchallenge.net>]

## Frameworks

* py-motmetrics[<https://github.com/cheind/py-motmetrics>] for evaluating trackers.

## Setup

1. Install [Cuda 9.0](https://developer.nvidia.com/cuda-90-download-archive)

    ```sh
    Do you accept the previously read EULA?
    accept/decline/quit: accept

    You are attempting to install on an unsupported configuration. Do you wish to continue?
    (y)es/(n)o [ default is no ]: y

    Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?
    (y)es/(n)o/(q)uit: n

    Install the CUDA 9.0 Toolkit?
    (y)es/(n)o/(q)uit: y

    Enter Toolkit Location
    [ default is /usr/local/cuda-9.0 ]:

    Do you want to install a symbolic link at /usr/local/cuda?
    (y)es/(n)o/(q)uit: n

    Install the CUDA 9.0 Samples?
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

2. Make sure that Cuda 9.0 is used

    ```sh
    $ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2017 NVIDIA Corporation
    Built on Fri_Sep__1_21:08:03_CDT_2017
    Cuda compilation tools, release 9.0, V9.0.176
    ```

    If the different version of nvcc is displayed, try to install a symbolic link

    ```sh
    sudo unlink /usr/local/cuda
    sudo ln -s /usr/local/cuda-9.0 /usr/local/cuda
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
    git clone https://github.com/songheony/deepmot-rev.git deepmot
    git clone https://github.com/shijieS/SST
    git clone https://github.com/bochinski/iou-tracker
    git clone --recurse-submodules --branch iccv_19 https://github.com/phil-bergmann/tracking_wo_bnw
    git clone https://github.com/nwojke/deep_sort
    git clone https://github.com/abewley/sort
    git clone https://github.com/longcw/MOTDT
    ```

2. Install necessary libraries with Anaconda 3

    ```sh
    conda create -n [ENV_NAME] python=3.6
    conda activate [ENV_NAME]
    # For parsing det.txt file
    conda install pandas black flake8
    # For DeepMOT and Tracktor, it is needed to install pytorch <= 0.4.1
    # For FPN which is used in Tracktor, some error occurs with pytorch==0.4.1
    conda install pytorch=0.4.0 torchvision cuda90 -c pytorch
    # For DeepSORT, it is needed to install tensorflow <= 2.0
    conda install tensorflow-gpu==1.14.0

    conda install pillow pyyaml cython matplotlib
    conda install -c conda-forge opencv easydict
    ```

3. Build FPN and FRCNN from Tracktor

    ```sh
    cd external/tracking_wo_bnw
    conda activate [ENV_NAME]
    bash src/fpn/fpn/make.sh
    bash src/frcnn/frcnn/make.sh
    ```

## How to run

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
