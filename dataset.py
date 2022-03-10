from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import config

class MyDataset(Dataset):
    def __init__(self, root_A, root_B):
        self.root_A = root_A
        self.root_B = root_B

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        A_img = config.transform_train(image=A_img)["image"]
        B_img = config.transform_train(image=B_img)["image"]

        return A_img, B_img

class MyTestDataset(Dataset):
    def __init__(self, root_A):
        self.root_A = root_A
        self.A_images = os.listdir(root_A)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.A_len

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        A_path = os.path.join(self.root_A, A_img)
        A_img = np.array(Image.open(A_path).convert("RGB"))
        A_img = config.transform_test(image=A_img)["image"]
        return A_img, A_path