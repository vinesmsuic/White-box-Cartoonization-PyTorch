# White-box-Cartoonization (PyTorch)

Unofficial PyTorch implementation of White-box-Cartoonization. We followed the original Tensorflow training implementation from the paper author ([Xinrui Wang](https://github.com/SystemErrorWang/White-box-Cartoonization)).

**Key difference from Tensorflow implementation:**
* We used PyTorchVGG19 instead of CaffeVGG model, which has a different range of input/output and std/mean.


![](https://github.com/SystemErrorWang/White-box-Cartoonization/raw/master/images/method.jpg)

# Our Results (Demo videos, click on the image)

[![Watch the video](https://img.youtube.com/vi/7qog3Dn4Bug/0.jpg)](https://youtu.be/7qog3Dn4Bug)

[![Watch the video](https://img.youtube.com/vi/4u3ol6Csvqg/0.jpg)](https://youtu.be/4u3ol6Csvqg)

# Repo Structure

```python
├─checkpoints
│  └─project_name
├─data
│  ├─train 
│  │  ├─cartoon # You put cartoon images here
│  │  └─photo   # You put photo images here
│  └─val
│      └─photo # You put photo images here
└─results
    ├─.... # folder will be created automatically
```


# Dependencies

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
4. Training (if you need to use the parser, type `python train.py -h` to see existing options

```bash
python train.py
```

* The training consist of initialization phase and training phase.
* Wait for a long time and see the results at `results` folder

# More options:

## Test
```
usage: test.py [-h] [--dataroot DATAROOT] [--weight_path WEIGHT_PATH] [--dest_folder DEST_FOLDER] [--sample_size SAMPLE_SIZE] [--shuffle] [--concat_img]
               [--no_post_processing]

test.py: Model testing script of White-box Cartoonization. For inference, please refer to inference.py

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to image data test folder. default path:data\val\photo
  --weight_path WEIGHT_PATH
                        path to model weight file. default path:checkpoints\project_name\i_gen.pth.tar
  --dest_folder DEST_FOLDER
                        path to destination folder for saving images. default path:results\project_name\test
  --sample_size SAMPLE_SIZE
                        only inference certain number of images. default=50.
  --shuffle             shuffle test data
  --concat_img          concat input and output images instead of separated save files
  --no_post_processing  disable post_processing (not recommended). This will probably cause output to have terrible noise
```

## Inference (Support Video)
```
usage: inference.py [-h] -s SOURCE -w WEIGHT_PATH [--batch_size BATCH_SIZE]

inference.py: Model inference script of White-box Cartoonization.

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        filepath to a source image or a video or a images folder.
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        path to model weight file.
  --batch_size BATCH_SIZE
                        batch size for video inference. default size:1
```



# TODO
* ~~[] Automatic Mixed Precision~~
* ~~[] LR Scheduler~~
* [] Loss visualization
* [] WandB visualization
* [] Adding Face data for Training
* [x] Parser
* [x] Post processing
* [x] Inference Code
* [x] Explaining Code

# Working Environments
* Windows with CUDA
* Ubuntu with CUDA
