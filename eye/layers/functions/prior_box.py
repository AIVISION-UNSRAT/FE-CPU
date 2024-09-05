import torch
from itertools import product as product
import numpy as np


class PriorBox2(object):
    def __init__(self, cfg2, box_dimension=None, image_size=None, phase='train'):
        super(PriorBox2, self).__init__()
        self.variance = cfg2['variance']
        self.min_sizes = cfg2['min_sizes']
        self.steps = cfg2['steps']
        self.aspect_ratios = cfg2['aspect_ratios']
        self.clip = cfg2['clip']
        if phase == 'train':
            self.image_size = (cfg2['min_dim'], cfg2['min_dim'])
            self.feature_maps = cfg2['feature_maps']
        elif phase == 'test':
            self.feature_maps = box_dimension.cpu().numpy().astype(np.int)
            self.image_size = image_size
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    mean += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
