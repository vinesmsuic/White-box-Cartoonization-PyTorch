import torch
import torch.nn as nn
import torch.optim as optim
import config
import os
import cv2
from typing import Generator, List, Tuple
from generator_model import Generator
import argparse
from utils import load_checkpoint
from more_itertools import chunked
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from surface_extractor import GuidedFilter

def parser():
    parser = argparse.ArgumentParser(description="inference.py: Model inference script of White-box Cartoonization.")
    parser.add_argument('-s',"--source",required=True,
                        help="filepath to a source image or a video or a images folder.")
    parser.add_argument('-w',"--weight_path", required=True,
                        help="path to model weight file.")
    parser.add_argument("--batch_size", type=int, default= config.BATCH_SIZE,
                        help="batch size for video inference. default size:"+f"{config.BATCH_SIZE}")
    parser.add_argument("--dest_folder", type=str, required=True,
                        help="Destination folder path for saving results.")
    parser.add_argument("--suffix", type=str, default= "_infered",
                        help="Output suffix.")
    return parser.parse_args()

def check_arguments_errors(args):
    if(check_format(args.source) == None):
        raise(ValueError("Invalid input path {}".format(os.path.abspath(args.source))))
    if not os.path.isfile(args.weight_path):
        raise(ValueError("Invalid model weight path {}".format(os.path.abspath(args.weight_path))))

def check_format(source_path):
    if(os.path.isdir(source_path)):
        return 'folder'
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
    VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
    if(source_path.split(".")[-1] in IMG_FORMATS):
        return 'image'
    if(source_path.split(".")[-1] in VID_FORMATS):
        return 'video'

    return None

def get_scaled_dim(height, width):
    height, width = reduce_to_scale([height, width], [config.IMAGE_SIZE, config.IMAGE_SIZE], scale=32)
    return height, width

def reduce_to_scale(img_hw: List[int], min_hw: List[int], scale: int) -> Tuple[int]:
    im_h, im_w = img_hw
    if im_h <= min_hw[0]:
        im_h = min_hw[0]
    else:
        x = im_h % scale
        im_h = im_h - x

    if im_w < min_hw[1]:
        im_w = min_hw[1]
    else:
        y = im_w % scale
        im_w = im_w - y
    return (im_h, im_w)

def get_transform(height, width):
    height, width = get_scaled_dim(height=height, width=width)
    transform_inference = A.Compose(
        [   
            A.Resize(width=width, height=height),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ]
    )
    return transform_inference

#===============================================
#NOTE Equation 9 in the paper, adjust sharpness of output
# If no post_processing, noise will be found
def post_processing(x, G_x, r: int=1):
    extract_surface = GuidedFilter()
    return extract_surface.process(x, G_x, r=r)
#===============================================

def infer_batch(img, model):
    img = img.to(config.DEVICE)
    out = model.forward(img)
    out = torch.tanh(out)
    out = post_processing(img, out)

    #TODO fix this later
    unnormalized_out = out*0.5+0.5

    bgr_img = (unnormalized_out.permute((0,2,3,1)).detach().to('cpu').numpy()*255).astype('uint8')
    return bgr_img

def infer_one_image(source_path, output_path, model, suffix):
    rgb_image = cv2.cvtColor(cv2.imread(source_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]
    transform_inference = get_transform(height=height, width=width)
    norm_image = transform_inference(image=rgb_image)["image"]
    out_img = infer_batch(norm_image[None, ...], model)
    out_img = out_img[0]
    cv2.imwrite(output_path+suffix+'.png', cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

#reference: https://github.com/zhen8838/AnimeStylized/blob/main/utils/video.py
def get_video_reader(path):
    video_cap = cv2.VideoCapture(path)
    length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def gen(cap):
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                cap.release()
                break
    return gen(video_cap), length, fps, height, width
                
def get_video_writer(path, fps, height, width):
    video_writer = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(width, height))
    return video_writer


def infer_video(source_path, output_path, batch_size, model, suffix):
    video_read, length, fps, height, width = get_video_reader(source_path)
    
    #update scaled height and width
    height, width = get_scaled_dim(height=height, width=width)

    transform_inference = get_transform(height=height, width=width)
    
    output_path = output_path+suffix+'.mp4'
    video_write = get_video_writer(output_path, fps, height, width)
    for frames in tqdm(chunked(video_read, batch_size), total=length // batch_size):
        norm_imgs = torch.stack(
            [transform_inference(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))["image"] for frame in frames]
        )
        output_imgs = infer_batch(norm_imgs, model)
        for img in output_imgs:
            video_write.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video_write.release()

def infer_fn(args):
    print("Using Device: " + config.DEVICE)
    gen = Generator(img_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(gen, opt_gen, config.LEARNING_RATE, path=args.weight_path)
    gen.eval()

    if(check_format(args.source) == 'video'):
        infer_video(args.source, args.source, batch_size=args.batch_size, model=gen, suffix=args.suffix)
    elif(check_format(args.source) == 'image'):
        infer_one_image(args.source, args.source, model=gen, suffix=args.suffix)
    
    if(check_format(args.source) == 'folder'):
        #Create Inference folder
        dest_folder = args.dest_folder
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        list_files = os.listdir(args.source)
        print(f'{len(list_files)} files found in the folder {args.source}. Output folder: {dest_folder}')

        for file in list_files:
            input_path = os.path.join(args.source,file)
            output_path = os.path.join(dest_folder,file)
            if(check_format(input_path) == 'video'):
                infer_video(input_path, output_path, batch_size=args.batch_size, model=gen, suffix=args.suffix)
                print(f"Finished inferencing file: {file}")
            elif(check_format(input_path) == 'image'):
                infer_one_image(input_path, output_path, model=gen, suffix=args.suffix)
                print(f"Finished inferencing file: {file}")
        
    print("=> Finish Inference.")

if __name__ == "__main__":
    args = parser()
    check_arguments_errors(args)
    infer_fn(args)

