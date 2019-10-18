
import numpy as np
from scipy.misc import imread, imresize


def load_image(filename, size=512):
    """Load and resize an image from disk.
    Inputs:
    :filename: path to file
    :size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    orig_shape = np.array(img.shape[:2])
    min_idx = np.argmin(orig_shape)
    scale_factor = float(size) / orig_shape[min_idx]
    new_shape = (orig_shape * scale_factor).astype(int)
    img = imresize(img, scale_factor)

    return img

