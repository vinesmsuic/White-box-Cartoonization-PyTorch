# White-box-Cartoonization (PyTorch)

Unofficial PyTorch implementation of White-box-Cartoonization. We followed the original Tensorflow training implementation from the paper author ([Xinrui Wang](https://github.com/SystemErrorWang/White-box-Cartoonization)).

![](https://github.com/SystemErrorWang/White-box-Cartoonization/raw/master/images/method.jpg)

# Repo Structure

```python
├─checkpoints
├─data
│  ├─train 
│  │  ├─cartoon # You put cartoon images here
│  │  └─photo   # You put photo images here
│  └─val
│      └─photo # You put photo images here
└─results
    ├─test  
    └─train
```


# Dependenies

* Albumentations

```bash
pip install -U albumentations
```

* tqdm

```bash
pip install tqdm
```

# To start training

0. Read https://vinesmsuic.github.io/2022/01/21/i2i-wbcartoonization to understand the implementation
1. Prepare the photo and cartoon data
2. Get the pre-trained VGG19 weight and put it in the root folder : 
   https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
3. Edit `config.py`
4. Training

```bash
python train.py
```

* The training consist of initialization phase and training phase.
* Wait for a long time and see the results at `results` folder


# TODO

* [] Automatic Mixed Precision
* [] LR Scheduler
* [] Loss visualization
* [] WandB visualization
* [] Inference Code
* [x] Explaining Code

# Working Environments

* Windows with CUDA
* Ubuntu with CUDA
