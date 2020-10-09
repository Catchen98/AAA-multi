# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound for Multiple Object Tracking

## Experts

* [CenterTrack](https://arxiv.org/abs/2004.01177)[<https://github.com/xingyizhou/CenterTrack>]
* [DAN](https://arxiv.org/abs/1810.11780)[<https://github.com/shijieS/SST>]
* [DeepMOT](https://arxiv.org/abs/1906.06618)[<https://gitlab.inria.fr/yixu/deepmot>]
* [Tracktor](https://arxiv.org/abs/1903.05625)[<https://github.com/phil-bergmann/tracking_wo_bnw>]
* [UMA](https://arxiv.org/abs/2003.11291)[<https://github.com/yinjunbo/UMA-MOT>]

## Feedback

* [MPNTracker](https://arxiv.org/abs/1912.07515)[https://github.com/dvl-tum/mot_neural_solver]

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
    # CenterTrack
    git clone https://github.com/xingyizhou/CenterTrack.git
    # DAN
    git clone https://github.com/shijieS/SST.git
    # DeepMOT
    git clone https://gitlab.inria.fr/yixu/deepmot.git
    # SORT
    git clone https://github.com/abewley/sort.git
    # Tracktor
    git clone https://github.com/phil-bergmann/tracking_wo_bnw.git
    # UMA
    git clone https://github.com/yinjunbo/UMA-MOT.git
    # MPNTracker
    git clone https://github.com/songheony/mot_neural_solver.git
    ```

2. Install necessary libraries with Anaconda 3

    For our framework

    ```sh
    conda create -n [ENV_NAME] python=3.6
    conda activate [ENV_NAME]

    # For AAA
    conda install -y black flake8 pandas seaborn
    conda install -y pytorch torchvision cudatoolkit=[CUDA_VERSION] -c pytorch
    pip install opencv-python

    # For feedback
    # Please refer https://github.com/rusty1s/pytorch_geometric to install torch-scatter, torch-sparse, torch-geometric.
    pip install torch-scatter==latest+[CUDA_VERSION] -f https://pytorch-geometric.com/whl/torch-[TORCH_VERSION].html
    pip install torch-sparse==latest+[CUDA_VERSION] -f https://pytorch-geometric.com/whl/torch-[TORCH_VERSION].html
    pip install torch-geometric
    pip install pytorch-lightning pulp
    conda install scikit-image

    # For evaluation
    pip install motmetrics lapsolver
    ```

    For CenterTrack

    ```sh
    cd external/CenterTrack
    conda create --name CenterTrack python=3.6
    conda activate CenterTrack
    conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=[CUDA_VERSION] -c pytorch
    pip install -r requirements.txt

    # For DCNv2
    cd src/lib/model/networks
    git clone https://github.com/CharlesShang/DCNv2.git
    cd DCNv2
    ./make.sh
    ```

    For DAN

    ```sh
    cd external/SST
    conda create --name DAN python=3.6
    conda activate DAN
    conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch

    pip install opencv-python==3.4.0.12 PyYAML==3.12 matplotlib

    # For our framework
    conda install -y pandas
    pip install motmetrics
    ```

    For DeepMOT

    ```sh
    cd external/deepMOT
    conda create --name DeepMOT python=3.6
    conda activate DeepMOT
    conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch
    pip install opencv-python==4.0.1.* PyYAML==4.2b1 easydict matplotlib

    # For our framework
    conda install -y pandas
    pip install motmetrics
    ```

    For DeepSORT

    ```sh
    cd external/deep_sort
    conda create --name DeepSORT python=3.6
    conda activate DeepSORT
    conda install -y tensorflow-gpu==1.14.0
    conda install -y scikit-learn pillow
    pip install opencv-python

    # For our framework
    conda install -y pandas pyyaml
    pip install motmetrics
    ```

    For GCNNMatch

    ```sh
    cd external/GCNNMatch
    conda create --name GCNNMatch python=3.7
    conda activate GCNNMatch
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    conda install tensorflow-gpu
    pip install opencv-python==4.0.1.24 torch-geometric scipy matplotlib Pillow
    pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
    pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html

    # For our framework
    conda install -y pyyaml pandas
    pip install motmetrics
    ```

    For MOTDT

    ```sh
    cd external/MOTDT
    conda create --name MOTDT python=3.6
    conda activate MOTDT
    conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch
    pip install opencv-python Cython scipy==1.1.0 numba scikit-learn=0.22.2 h5py

    # For our framework
    conda install -y pyyaml pandas
    pip install motmetrics
    ```

    For Tracktor

    ```sh
    cd external/tracking_wo_bnw
    conda create --name Tracktor python=3.6
    conda activate Tracktor
    conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=[CUDA_VERSION] -c pytorch
    pip install opencv-python==4.0.1.24 cycler matplotlib

    # For our framework
    conda install -y pyyaml pandas
    pip install motmetrics
    ```

    For UMA

    ```sh
    cd external/UMA-MOT
    conda create --name UMA python=3.6
    conda activate UMA
    conda install -y tensorflow-gpu==1.14.0
    pip install -r requirements.txt

    # For Tracktor
    conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=[CUDA_VERSION] -c pytorch
    pip install cycler opencv-python==4.0.1.24

    # For our framework
    conda install -y pyyaml pandas
    pip install motmetrics
    ```

## How to run

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
