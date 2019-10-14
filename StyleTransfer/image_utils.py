
import numpy as np
from scipy.misc import imread, imresize


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img):
    """
    Preprocess an image for SqueezeNet.
    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    img = np.array(img)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


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

