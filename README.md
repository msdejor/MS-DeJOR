# MS-DeJoR: A Hybrid Approach for Webly-supervised Fine-Grained Visual Classification
## Introduction
This is the PyTorch implementation for our paper **MS-DeJoR: A Hybrid Approach for Webly-supervised Fine-Grained Visual Classification**

## Network Architecture
The architecture of our proposed approach is as follows
<img src="architecture.png" width="100%" />


## Environment
Create a virtual environment with python 3.7,

    $  conda create -n msdejor_env python=3.7

    $  conda activate msdejor_env

  Install all dependencies

    $  pip install -r requirements.txt

## Data Preparation
Download these web fine-grained datasets, namely [Web-CUB](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz), [Web-Car](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz) and [Web-Aircraft](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz). Then uncompress them into `./data` directory.

  ```
  data
  ├── web-bird
  │   ├── train
  │   └── val
  ├── web-car
  │   ├── train
  │   └── val
  └── web-aircraft
      ├── train
      └── tval
  ```

## Running
- If you want to use multi-scale module,  modify the corresponding parameters in `main_msdejor.py` or directly run `main_msdejor.py` to get the final result. We provide the default parameter settings as following：

```python
python main_msdejor.py --bs 30 --net 50 --data bird --lamb 0.1 --gama 2  
```

- If you only prefer the DeJoR module,  run `main_dejor.py`.

```python
python main_dejor.py --bs 50 --net 18 --data bird --lamb 0.1 --gama 2
```
In our experiments, we adopt the same hyperparameters across three benchmark datasets, setting $\lambda$ to 0.1 and $\gamma$ to 2.



