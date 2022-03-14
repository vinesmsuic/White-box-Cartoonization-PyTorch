import numpy as np
from skimage import segmentation
from skimage.feature import local_binary_pattern
from scipy.ndimage import find_objects
from skimage.segmentation import find_boundaries
from skimage.color import rgb2lab
from joblib import Parallel, delayed
import torch
import torch.nn as nn

class SuperPixel():
    def __init__(self, device: torch.device='cpu', mode='simple'):
        self.device = device
        self.mode = mode

    def process(self, x: torch.Tensor):
        # B, C, H, W => B, H, W, C
        # Torch => Numpy
        skimage_format_tensor = x.permute((0, 2, 3, 1)).cpu().numpy()

        if(self.mode == 'simple'):
            skimage_format_tensor = simple_superpixel(skimage_format_tensor)
        elif(self.mode == 'sscolor'):
            skimage_format_tensor = selective_adacolor(skimage_format_tensor)

        # B, H, W, C => B, C, H, W
        # Numpy => Torch
        return torch.from_numpy(skimage_format_tensor).to(self.device).permute((0, 3, 1, 2))

# Adaptive Coloring
def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        color: np.ndarray = None
        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std < 20:
                color = image[mask].mean(axis=0)
            elif 20 < std < 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5 * mean + 0.5 * median
            elif 40 < std:
                color = np.median(image[mask], axis=0)
        out[mask] = color
    return out

# Simple Linear Iterative Clustering
def slic(image, seg_num=200, kind='mix'):
    seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,compactness=10, convert2lab=True)
    image = label2rgb(seg_label, image, kind=kind, bg_label=-1)
    return image

# Apply slic to batches
def simple_superpixel(batch_image, seg_num=200, kind='mix'):
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(slic)\
                         (image, seg_num, kind) for image in batch_image)
    return np.array(batch_out)

# Felzenszwalb algorithm + Selective Search
def color_ss_map(image, seg_num=200, power=1.2, k=10, sim_strategy='CTSF'):
    
    img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    
    img_cvtcolor = rgb2lab(img_cvtcolor)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping
    while S.num_regions() > seg_num:
        
        i,j = S.get_highest_similarity()
        S.merge_region(i,j)
        S.remove_similarities(i,j)
        S.calculate_similarity_for_new_region()
    
    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image+1)/2
    image = image**power
    if(not np.max(image)==0):
        image = image/np.max(image)
    image = image*2 - 1
    return image

# Apply color_ss_map to batches
def selective_adacolor(batch_image, seg_num=200, power=1.2):
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(color_ss_map)\
                         (image, seg_num, power) for image in batch_image)
    return np.array(batch_out)


class HierarchicalGrouping(object):
    def __init__(self, img, img_seg, sim_strategy):
        self.img = img
        self.sim_strategy = sim_strategy
        self.img_seg = img_seg.copy()
        self.labels = np.unique(self.img_seg).tolist()

    def build_regions(self):
        self.regions = {}
        lbp_img = generate_lbp_image(self.img)
        for label in self.labels:
            size = (self.img_seg == 1).sum()
            region_slice = find_objects(self.img_seg==label)[0]
            box = tuple([region_slice[i].start for i in (1,0)] +
                         [region_slice[i].stop for i in (1,0)])

            mask = self.img_seg == label
            color_hist = calculate_color_hist(mask, self.img)
            texture_hist = calculate_texture_hist(mask, lbp_img)

            self.regions[label] = {
                'size': size,
                'box': box,
                'color_hist': color_hist,
                'texture_hist': texture_hist
            }

    def build_region_pairs(self):
        self.s = {}
        for i in self.labels:
            neighbors = self._find_neighbors(i)
            for j in neighbors:
                if i < j:
                    self.s[(i,j)] = calculate_sim(self.regions[i],
                                             self.regions[j],
                                             self.img.size,
                                             self.sim_strategy)

    def _find_neighbors(self, label):
        """
            Parameters
        ----------
            label : int
                label of the region
        Returns
        -------
            neighbors : list
                list of labels of neighbors
        """

        boundary = find_boundaries(self.img_seg == label,
                                   mode='outer')
        neighbors = np.unique(self.img_seg[boundary]).tolist()

        return neighbors

    def get_highest_similarity(self):
        return sorted(self.s.items(), key=lambda i: i[1])[-1][0]

    def merge_region(self, i, j):

        # generate a unique label and put in the label list
        new_label = max(self.labels) + 1
        self.labels.append(new_label)

        # merge blobs and update blob set
        ri, rj = self.regions[i], self.regions[j]

        new_size = ri['size'] + rj['size']
        new_box = (min(ri['box'][0], rj['box'][0]),
                  min(ri['box'][1], rj['box'][1]),
                  max(ri['box'][2], rj['box'][2]),
                  max(ri['box'][3], rj['box'][3]))
        value = {
            'box': new_box,
            'size': new_size,
            'color_hist':
                (ri['color_hist'] * ri['size']
                + rj['color_hist'] * rj['size']) / new_size,
            'texture_hist':
                (ri['texture_hist'] * ri['size']
                + rj['texture_hist'] * rj['size']) / new_size,
        }

        self.regions[new_label] = value

        # update segmentation mask
        self.img_seg[self.img_seg == i] = new_label
        self.img_seg[self.img_seg == j] = new_label

    def remove_similarities(self, i, j):

        # mark keys for region pairs to be removed
        key_to_delete = []
        for key in self.s.keys():
            if (i in key) or (j in key):
                key_to_delete.append(key)

        for key in key_to_delete:
            del self.s[key]

        # remove old labels in label list
        self.labels.remove(i)
        self.labels.remove(j)

    def calculate_similarity_for_new_region(self):
        i = max(self.labels)
        neighbors = self._find_neighbors(i)

        for j in neighbors:
            # i is larger than j, so use (j,i) instead
            self.s[(j,i)] = calculate_sim(self.regions[i],
                                          self.regions[j],
                                          self.img.size,
                                          self.sim_strategy)

    def is_empty(self):
        return True if not self.s.keys() else False
    
    
    def num_regions(self):
        return len(self.s.keys())

def calculate_color_hist(mask, img):
    """
        Calculate colour histogram for the region.
        The output will be an array with n_BINS * n_color_channels.
        The number of channel is varied because of different
        colour spaces.
    """

    BINS = 25
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    channel_nums = img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist


def generate_lbp_image(img):

    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    channel_nums = img.shape[2]

    lbp_img = np.zeros(img.shape)
    for channel in range(channel_nums):
        layer = img[:, :, channel]
        lbp_img[:, :,channel] = local_binary_pattern(layer, 8, 1)

    return lbp_img


def calculate_texture_hist(mask, lbp_img):
    """
        Use LBP for now, enlightened by AlpacaDB's implementation.
        Plan to switch to Gaussian derivatives as the paper in future
        version.
    """

    BINS = 10
    channel_nums = lbp_img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = lbp_img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist


def calculate_sim(ri, rj, imsize, sim_strategy):
    """
        Calculate similarity between region ri and rj using diverse
        combinations of similarity measures.
        C: color, T: texture, S: size, F: fill.
    """
    sim = 0

    if 'C' in sim_strategy:
        sim += _calculate_color_sim(ri, rj)
    if 'T' in sim_strategy:
        sim += _calculate_texture_sim(ri, rj)
    if 'S' in sim_strategy:
        sim += _calculate_size_sim(ri, rj, imsize)
    if 'F' in sim_strategy:
        sim += _calculate_fill_sim(ri, rj, imsize)

    return sim

def _calculate_color_sim(ri, rj):
    """
        Calculate color similarity using histogram intersection
    """
    return sum([min(a, b) for a, b in zip(ri["color_hist"], rj["color_hist"])])


def _calculate_texture_sim(ri, rj):
    """
        Calculate texture similarity using histogram intersection
    """
    return sum([min(a, b) for a, b in zip(ri["texture_hist"], rj["texture_hist"])])


def _calculate_size_sim(ri, rj, imsize):
    """
        Size similarity boosts joint between small regions, which prevents
        a single region from engulfing other blobs one by one.
        size (ri, rj) = 1 − [size(ri) + size(rj)] / size(image)
    """
    return 1.0 - (ri['size'] + rj['size']) / imsize


def _calculate_fill_sim(ri, rj, imsize):
    """
        Fill similarity measures how well ri and rj fit into each other.
        BBij is the bounding box around ri and rj.
        fill(ri, rj) = 1 − [size(BBij) − size(ri) − size(ri)] / size(image)
    """

    bbsize = (max(ri['box'][2], rj['box'][2]) - min(ri['box'][0], rj['box'][0])) * (max(ri['box'][3], rj['box'][3]) - min(ri['box'][1], rj['box'][1]))

    return 1.0 - (bbsize - ri['size'] - rj['size']) / imsize


if __name__ == "__main__":
    super_pixel = SuperPixel(mode='sscolor')
    input = torch.randn(5,3,256,256)
    result = super_pixel.process(input)
    print(result.shape)