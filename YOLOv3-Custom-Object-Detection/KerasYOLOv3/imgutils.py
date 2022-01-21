from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin_to_clr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    # ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    image_bin = cv2.threshold(image_gs, 0, 255,
					cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def remove_noise(image):
    return erode(dilate(image))

def get_binary_remove_noise(image):
    image_binary = image_bin(image_gray(image))
    return remove_noise(image_binary)

def get_binary(image):
    return image_bin(image_gray(image))

def scale_image(image, scale=1.):
    return cv2.resize(
            image, 
            (int(image.shape[1]/scale), int(image.shape[0]//scale)), 
            interpolation=cv2.INTER_CUBIC
        )

def resize_image(image, width, height):
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

def convert_image_float(image):
    image = image.astype('float32')
    image /= 255.
    return image

def load_image_resize(path, width=416, height=416):
    image = load_image(path)
    original_height, original_width  = image.shape[0], image.shape[1]
    image = resize_image(image, width, height)
    image = convert_image_float(image)
    image = expand_dims(image, 0)
    return image, original_width, original_height


