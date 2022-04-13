import numpy as np
from torchvision import transforms


class PairGenerator(object):
    def __init__(self, n_views=2):
        # TODO: more transforms
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([color_jitter], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor()])
        self.n_views = n_views

    def __call__(self, x):
        return [self.transforms(x) for i in range(self.n_views)]
