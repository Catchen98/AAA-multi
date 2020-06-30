# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound for Multiple Object Tracking

## Experts

* [DAN](https://arxiv.org/abs/1810.11780)[<https://github.com/shijieS/SST>]
* [DeepMOT](https://arxiv.org/abs/1906.06618)[<https://github.com/yihongXU/deepMOT>]
* [DeepSORT](https://arxiv.org/abs/1812.00442)[<https://github.com/nwojke/deep_sort>]
* [Deep-TAMA](https://arxiv.org/abs/1907.00831)[<https://github.com/yyc9268/Deep-TAMA>]
* [MOTDT](https://arxiv.org/abs/1809.04427)[<https://github.com/longcw/MOTDT>]
* [SORT](https://arxiv.org/abs/1602.00763)[<https://github.com/abewley/sort>]
* [Tracktor](https://arxiv.org/abs/1903.05625)[<https://github.com/phil-bergmann/tracking_wo_bnw>]

## Datasets

* [MOT16, MOT17](https://arxiv.org/abs/1603.00831)[<https://motchallenge.net>]
* [MOT20](https://arxiv.org/abs/2003.09003)[<https://motchallenge.net>]

## Frameworks

* py-motmetrics[<https://github.com/cheind/py-motmetrics>] for evaluating trackers.

## Requirements

1. Clone the repository of experts and feedback

    ```sh
    mkdir external
    cd external
    git clone https://github.com/shijieS/SST.git
    git clone --branch obsolete https://github.com/yihongXU/deepMOT.git
    git clone https://github.com/LeonLok/Deep-SORT-YOLOv4
    git clone https://github.com/yyc9268/Deep-TAMA
    git clone https://github.com/songheony/MOTDT
    git clone https://github.com/abewley/sort.git
    git clone https://github.com/phil-bergmann/tracking_wo_bnw.git
    git clone https://github.com/songheony/mot_neural_solver.git
    ```

2. Install necessary libraries with Anaconda 3

    ```sh
    conda create -n [ENV_NAME] python=3.6
    conda activate [ENV_NAME]

    # For our framework
    conda install -y black flake8 pandas
    conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch-nightly
    conda install -y tensorflow-gpu

    # For experts
    conda install -y pillow pyyaml cython matplotlib numba scikit-image scipy
    conda install -y -c conda-forge opencv filterpy

    # For feedback
    pip install torch-scatter torch-sparse pytorch-lightning pulp
    git clone https://github.com/rusty1s/pytorch_geometric.git external/pytorch_geometric
    cd pytorch_geometric
    python setup.py install

    # For evaluation
    pip install motmetrics lapsolver
    ```

## How to run

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
