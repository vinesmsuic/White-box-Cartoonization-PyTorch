import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy
import os
from torchvision.utils import save_image


def save_training_images(image, epoch, step, folder, suffix_filename:str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_image(image, os.path.join(folder, f"epoch_{epoch}_step_{step}_{suffix_filename}.png"))


def save_val_examples(gen, val_loader, epoch, step, folder, num_samples=1, concat_image=True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    gen.eval()
    with torch.no_grad():
        num_saved = 0
        for _, (x) in enumerate(val_loader):
            x = x.to(config.DEVICE)
            y_fake = gen(x)

            if(concat_image):
                save_image(torch.cat((x * 0.5 + 0.5,y_fake * 0.5 + 0.5), axis=3), os.path.join(folder, f"epoch_{epoch}_step_{step}_io{num_saved}.png"))
            else:
                save_image(y_fake * 0.5 + 0.5, os.path.join(folder, f"epoch_{epoch}_step_{step}_gen{num_saved}.png"))
                save_image(x * 0.5 + 0.5, os.path.join(folder, f"epoch_{epoch}_step_{step}_input{num_saved}.png"))

            num_saved += 1
            if(num_saved == num_samples):
                break
    gen.train()

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


def load_checkpoint(model, optimizer, lr, folder, checkpoint_file):
    print("=> Loading checkpoint")
    path = os.path.join(folder, checkpoint_file)
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
