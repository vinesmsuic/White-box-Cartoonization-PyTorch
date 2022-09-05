import gradio as gr

import torch
import torch.optim as optim
import torchvision.transforms as T

import config
import os

import numpy as np
from generator_model import Generator
import argparse
from utils import load_checkpoint
from inference import get_transform, post_processing

parser = argparse.ArgumentParser(description="")
parser.add_argument('-w',"--weight_path", required=True,
                        help="path to model weight file.")
opt = parser.parse_args()

weight_path = opt.weight_path

gen = Generator(img_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
load_checkpoint(gen, opt_gen, config.LEARNING_RATE, path=weight_path)
gen.eval()

def infer_one_image(pil_image, model, post_r=1):
    width, height = pil_image.size
    transform_inference = get_transform(height=height, width=width)
    # Reshape image
    img = np.array(pil_image)
    img = transform_inference(image=img)["image"]
    # Turn 3-dim image into 4-dim tensor
    img = torch.unsqueeze(img, dim=0)
    img = img.to(config.DEVICE)
    out = model(img)
    out = torch.tanh(out)
    out = post_processing(img, out, post_r)
    # Unnormalize image
    unnormalized_out = out*0.5+0.5
    return unnormalized_out

def torch_to_pil(torch_image):
    torch_image = torch.squeeze(torch_image, dim=0)
    PIL_transform = T.ToPILImage()
    return PIL_transform(torch_image)

def cartoonize(image, post_processing_r):
    output = infer_one_image(image, gen, post_processing_r)
    return torch_to_pil(output)


def get_examples(path):
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'JPEG', 'PNG', 'JPG'  # include image suffixes
    list_img_relpath = []
    for img in os.listdir(path):
        if (img.split(".")[-1] in IMG_FORMATS) ==True:
            list_img_relpath.append([os.path.join(path, img), 1])
    return list_img_relpath


if __name__ == "__main__":
    sample_path = "demo_examples"
    list_img = get_examples(sample_path)
    print(list_img)

    demo = gr.Interface(
    fn=cartoonize,
    inputs=[gr.Image(type="pil"), gr.Slider(0, 10, 1, step=1)],
    outputs=[gr.Image(type="pil")],
    examples=list_img
    )
    demo.launch()

