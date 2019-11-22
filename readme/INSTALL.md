# Installation


The code was tested on Ubuntu 18.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.1 with CUDA 10.0. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CenterNet python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet
    ~~~

1. Install pytorch1.1:

    ~~~
    conda install pytorch=1.1 torchvision cuda100 -c pytorch
    ~~~
    
    And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
         
     For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 
     
2. Install [COCOAPI]:

    ~~~
    pip install pycocotools
    ~~~

3. Clone this repo:

    ~~~
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2)).

    ~~~
    cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
    conda install conda
    conda install -c conda-forge cudatoolkit-dev=10.0
    python setup.py build develop
    ~~~
6. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet. 

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    python setup.py build_ext --inplace
    ~~~

7. Download pertained models for [detection]() or [pose estimation]() and move them to `$CenterNet_ROOT/models/`. More models can be found in [Model zoo](MODEL_ZOO.md).
