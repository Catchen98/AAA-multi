# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound for Multiple Object Tracking

## Experts

* [DAN](https://arxiv.org/abs/1810.11780)[<https://github.com/shijieS/SST>]
* [DeepSORT](https://arxiv.org/abs/1812.00442)[<https://github.com/nwojke/deep_sort>]
* [IOU](https://ieeexplore.ieee.org/document/8078516)[<https://github.com/bochinski/iou-tracker>]
* [SORT](https://arxiv.org/abs/1602.00763)[<https://github.com/abewley/sort>]
* [Tracktor](https://arxiv.org/abs/1903.05625)[<https://github.com/phil-bergmann/tracking_wo_bnw>]
* [VIOU](https://ieeexplore.ieee.org/document/8639144)[<https://github.com/bochinski/iou-tracker>]

## Datasets

* [MOT16, MOT17](https://arxiv.org/abs/1603.00831)[<https://motchallenge.net>]
* [MOT20](https://arxiv.org/abs/2003.09003)[<https://motchallenge.net>]

## Frameworks

* py-motmetrics[<https://github.com/cheind/py-motmetrics>] for evaluating trackers.

## Requirements

1. Clone the repository of experts

    ```sh
    mkdir external
    cd external
    git clone https://github.com/shijieS/SST.git
    git clone https://github.com/nwojke/deep_sort.git
    git clone https://github.com/bochinski/iou-tracker.git
    git clone https://github.com/abewley/sort.git
    git clone --recurse-submodules https://github.com/phil-bergmann/tracking_wo_bnw.git
    ```

2. Install necessary libraries with Anaconda 3

    ```sh
    conda create -n [ENV_NAME] python=3.6
    conda activate [ENV_NAME]

    # For our framework
    conda install -y black flake8 pandas
    conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

    # For DeepSORT, it is needed to install tensorflow <= 2.0
    conda install -y tensorflow-gpu==1.14.0

    # For experts
    conda install -y pillow pyyaml cython matplotlib scikit-learn==0.22.1 scikit-image tqdm numba
    conda install -y -c conda-forge opencv easydict filterpy
    pip install motmetrics lapsolver
    ```

## How to run

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
