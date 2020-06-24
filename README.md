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

## Requirements

1. Clone the repository of experts

    ```sh
    mkdir external
    cd external
    git clone --branch obsolete https://gitlab.inria.fr/yixu/deepmot.git
    git clone https://github.com/shijieS/SST
    git clone https://github.com/bochinski/iou-tracker
    git clone --recurse-submodules https://github.com/phil-bergmann/tracking_wo_bnw
    git clone https://github.com/nwojke/deep_sort
    git clone https://github.com/abewley/sort
    git clone https://github.com/longcw/MOTDT
    ```

2. Install necessary libraries with Anaconda 3

    ```sh
    conda create -n [ENV_NAME] python=3.6
    conda activate [ENV_NAME]
    # For reformatting code
    conda install black flake8
    # For our framework
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    # For DeepSORT, it is needed to install tensorflow <= 2.0
    conda install tensorflow-gpu==1.14.0

    # For parsing det.txt file
    conda install pandas

    # For experts
    conda install pillow pyyaml cython matplotlib
    conda install -c conda-forge opencv easydict
    pip install motmetrics
    ```

## How to run

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
