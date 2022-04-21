import numpy as np
from torchvision import transforms


class PairGenerator(object):
    def __init__(self, size, n_views=2):
        # TODO: more transforms
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor()])
        self.n_views = n_views

    def __call__(self, x):
        return [self.transforms(x) for i in range(self.n_views)]
