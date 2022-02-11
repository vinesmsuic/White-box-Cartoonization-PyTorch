import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy
import os
from torchvision.utils import save_image

def save_test_examples(gen, val_loader, epoch, folder):
    x = next(iter(val_loader))
    x = x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, os.path.join(folder, f"y_gen_{epoch}.png"))
        save_image(x * 0.5 + 0.5, os.path.join(folder, f"input_{epoch}.png"))
    gen.train()

def save_checkpoint(model, optimizer, epoch, folder, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(folder, str(epoch) + "_" + filename)
    torch.save(checkpoint, path)


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

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)