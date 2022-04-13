import torch
import torch.nn as nn
import torch.optim as optim
import config
import os
from dataset import MyTestDataset
from generator_model import Generator
from utils import save_test_examples, load_checkpoint
import argparse

def parser():
    parser = argparse.ArgumentParser(description="test.py: Model testing script of White-box Cartoonization. For inference, please refer to inference.py")
    parser.add_argument("--dataroot",default=config.VAL_PHOTO_DIR,
                        help="path to image data test folder. default path:"+f"{config.VAL_PHOTO_DIR}")
    parser.add_argument("--weight_path", default=os.path.join(config.CHECKPOINT_FOLDER, config.LOAD_CHECKPOINT_GEN),
                        help="path to model weight file. default path:"+f"{os.path.join(config.CHECKPOINT_FOLDER, config.LOAD_CHECKPOINT_GEN)}")
    parser.add_argument("--dest_folder",default=config.RESULT_TEST_DIR,
                        help="path to destination folder for saving images. default path:"+f"{config.RESULT_TEST_DIR}")
    parser.add_argument("--sample_size", type=int, default= 50,
                        help="only inference certain number of images. default=50.")
    parser.add_argument('--shuffle', action='store_true',
                        help="shuffle test data")
    parser.add_argument('--concat_img', action='store_true',
                        help="concat input and output images instead of separated save files")
    parser.add_argument('--no_post_processing', action='store_true',
                        help="disable post_processing (not recommended). This will probably cause output to have terrible noise")
    return parser.parse_args()

def check_arguments_errors(args):
    if not os.path.isdir(args.dataroot):
        raise(ValueError("Invalid image data folder path {}".format(os.path.abspath(args.dataroot))))
    if not os.path.isfile(args.weight_path):
        raise(ValueError("Invalid model weight path {}".format(os.path.abspath(args.weight_path))))

def test_fn(args):
    print("Using Device: " + config.DEVICE)
    gen = Generator(img_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(gen, opt_gen, config.LEARNING_RATE, path=args.weight_path)
    test_dataset = MyTestDataset(args.dataroot)

    shuffle = True if(args.shuffle) else False
    concat_img = True if(args.concat_img) else False
    post_processing = False if(args.no_post_processing) else True
    
    print("=> Saving Test outputs")
    print("="*80)
    save_test_examples(gen, test_dataset, dest_folder=args.dest_folder, num_samples=args.sample_size, shuffle=shuffle, concat_image=concat_img, post_processing=post_processing)
    print(f"=> Finished inferencing {args.sample_size} images")
    print("="*80)

if __name__ == "__main__":
    args = parser()
    check_arguments_errors(args)
    test_fn(args)