import os
import random
import numpy as np
from PIL.Image import NEAREST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.core.fromnumeric import sort
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 
'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

letter_pics = os.listdir('character_data')
letter_labels = [classes.index(fname.split('_')[0]) for fname in letter_pics]
letter_pics = [os.path.join('character_data', fname) for fname in letter_pics]
sampled_letter_fnames = random.sample(letter_pics, 16)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    brightness_range=(0.2, 0.7),
    shear_range=0.2,
    zoom_range=0.15,
    fill_mode='nearest'
)
train_onehot = to_categorical(letter_labels)
IMG_DIM = (50, 30)
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in letter_pics]
train_imgs = np.array(train_imgs)
train_labels = np.array(letter_labels)

train_generator = train_datagen.flow(
    train_imgs,
    train_onehot,
    batch_size=32
)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape = (50, 30, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics = ['accuracy'])

history = model.fit_generator(train_generator,
                            steps_per_epoch=len(letter_pics)//32,
                            epochs=1000,
                            verbose=1)


model.save('char_model')

def get_predicted_label(prediction):
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    prediction_index = np.argmax(np.round(prediction), axis=1)          
    return classes[prediction_index]     