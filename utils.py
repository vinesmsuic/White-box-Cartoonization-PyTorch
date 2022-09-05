import random, torch, os, numpy as np
import torch.nn as nn
import config
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from surface_extractor import GuidedFilter

def save_training_images(image, epoch, step, dest_folder, suffix_filename:str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    save_image(image, os.path.join(dest_folder, f"epoch_{epoch}_step_{step}_{suffix_filename}.png"))


def save_val_examples(gen, val_loader, epoch, step, dest_folder, num_samples=1, concat_image=True, post_processing=True):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    gen.eval()
    with torch.no_grad():
        num_saved = 0
        for _, (x, img_path) in enumerate(val_loader):
            basename = os.path.basename(img_path[0]) # img_path[0]=> unpacking the tuple img_path
            basename = os.path.splitext(basename)[0] # remove extension name
            x = x.to(config.DEVICE)
            y_fake = gen(x)
            if(post_processing):
                y_fake = torch.tanh(y_fake)
                extract_surface = GuidedFilter()
                y_fake = extract_surface.process(x, y_fake, r=1)

            # * 0.5 + 0.5 is to remove the normalization used in config.py
            if(concat_image):
                save_image(torch.cat((x * 0.5 + 0.5,y_fake * 0.5 + 0.5), axis=3), os.path.join(dest_folder, f"epoch_{epoch}_step_{step}_io_{basename}.png"))
            else:
                save_image(y_fake * 0.5 + 0.5, os.path.join(dest_folder, f"epoch_{epoch}_step_{step}_gen_{basename}.png"))
                save_image(x * 0.5 + 0.5, os.path.join(dest_folder, f"epoch_{epoch}_step_{step}_input_{basename}.png"))

            num_saved += 1
            if(num_saved == num_samples):
                break
    gen.train()

def save_test_examples(gen, test_dataset, dest_folder, num_samples=50, shuffle=False, concat_image=False, post_processing=True):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=config.NUM_WORKERS)

    with torch.no_grad():
        for idx, (x, img_path) in enumerate(test_loader):
            basename = os.path.basename(img_path[0]) # img_path[0]=> unpacking the tuple img_path
            basename = os.path.splitext(basename)[0] # remove extension name
            if(idx >= num_samples):
                break
            x = x.to(config.DEVICE)
            y_fake = gen(x)
            if(post_processing):
                y_fake = torch.tanh(y_fake)
                extract_surface = GuidedFilter()
                y_fake = extract_surface.process(x, y_fake, r=1)
            if(concat_image):
                save_image(torch.cat((x * 0.5 + 0.5,y_fake * 0.5 + 0.5), axis=3), os.path.join(dest_folder, f"{basename}_io.png"))
            else:
                save_image(y_fake * 0.5 + 0.5, os.path.join(dest_folder, f"{basename}_gen.png"))
                save_image(x * 0.5 + 0.5, os.path.join(dest_folder, f"{basename}.png"))


def save_checkpoint(model, optimizer, epoch, folder, filename="my_checkpoint.pth.tar"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(folder, str(epoch) + "_" + filename)
    torch.save(checkpoint, path)
    print("=> checkpoint saved: " + str(path))


def load_checkpoint(model, optimizer, lr, path):
    print("=> Loading checkpoint")
    if (os.path.isfile(path)):
        checkpoint = torch.load(path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print("checkpoint file " + str(path) + " loaded.")
        loaded = True
    else:
        print("checkpoint file " + str(path) + " not found. Not loading checkpoint.")
        loaded = False
    return loaded

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bgr_cv2rgb(img):
    return img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB