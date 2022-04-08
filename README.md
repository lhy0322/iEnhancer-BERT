# iEnhancer-BERT
## 1. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). We applied training on a single NVIDIA Tesla V100 with 32 GB graphic memory. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

#### 1.1 Create and activate a new virtual environment

```
conda create -n ienhancer-bert python=3.6
conda activate ienhancer-bert
```



#### 1.2 Install the package and other requirements

(Required)

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
git clone https://github.com/lhy0322/iEnhancer-BERT
cd iEnhancer-BERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```
