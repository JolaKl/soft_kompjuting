import os
import random
import numpy as np
from PIL.Image import NEAREST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.engine.training_generator_v1 import evaluate_generator
from numpy.core.fromnumeric import sort
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
import cv2

from keras.models import load_model

from KerasYOLOv3 import imgutils
from KerasYOLOv3.contourutils import select_roi
from KerasYOLOv3.image import model_size, max_output_size, \
    max_output_size_per_class, iou_threshold, confidence_threshold, class_name
from KerasYOLOv3.imgutils import *
from KerasYOLOv3.utils import *

def convert_to_bin(val_imgs):
    bin_img = []
    for image in val_imgs:
        image = np.array(image, dtype='uint8')
        image = tf.expand_dims(image, axis=0)
        image = np.squeeze(image)
        char_image = imgutils.resize_image(image, 30, 50)
        char_image = imgutils.image_bin_to_clr(imgutils.invert(imgutils.get_binary_remove_noise(char_image)))
        imgutils.display_image(char_image)

        bin_img.append(char_image)
    return bin_img


classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
           'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

letter_pics = os.listdir('character_data_validation')
letter_labels = [classes.index(fname.split('_')[0]) for fname in letter_pics]
letter_pics = [os.path.join('character_data_validation', fname) for fname in letter_pics]
sampled_letter_fnames = random.sample(letter_pics, 16)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_labels_enc = to_categorical(letter_labels)
IMG_DIM = (50, 30)

validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in letter_pics]
validation_imgs = convert_to_bin(validation_imgs)

validation_imgs = np.array(validation_imgs)
test_labels = np.array(letter_labels)

validation_generator = validation_datagen.flow(
    validation_imgs,
    validation_labels_enc
)

model = load_model('char_model')


def process_image_file_v(img_path, model, debug=False, remove_noise=False):
    if not os.path.isfile(img_path):
        print(img_path + ' is not a file')
        return
    image = imgutils.load_image(img_path)
    print(img_path + ' :', end=' ')
    process_image_v(image, model, debug, remove_noise)
    print('\n-------------------------------------------')


def process_image_v(image, model, debug=False, remove_noise=False):
    image = np.array(image)
    image = tf.expand_dims(image, axis=0)
    # resized_frame = resize_image(image, (model_size[0], model_size[1]))

    # class_names = load_class_names(class_name)

    image = np.squeeze(image)

    char_image = imgutils.resize_image(image, 30, 50)
    char_image = imgutils.image_bin_to_clr(imgutils.invert(imgutils.get_binary_remove_noise(char_image)))
    imgutils.display_image(image)
    img_array = np.array([char_image])
    prediction = model.predict(img_array)
    predicted_char = get_predicted_label(prediction)
    print(predicted_char, end='')

    # imgutils.display_image(image, color=True)


def validation(path, model):
    letter_pics = os.listdir(path)
    for i in range(len(letter_pics)):
        process_image_file_v(path + letter_pics[i], model)


def get_predicted_label(prediction):
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    prediction_index = np.argmax(np.round(prediction), axis=1)[0]
    return classes[prediction_index]


def validate():
    print('Evaluation:')
    scoreSeg = model.evaluate(validation_generator)
    print("Accuracy = ", scoreSeg[1])


if __name__ == '__main__':
    show_predictions = False

    if show_predictions:
        validation('./character_data_validation/', model)

    validate()
