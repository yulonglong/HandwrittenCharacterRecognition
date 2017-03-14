import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_model():
    from keras.layers import Input, Flatten, Dense, Convolution2D, MaxPooling2D
    from keras.models import Model

    #Create your own input format (here 3x200x200)
    img_input = Input(shape=(16,8,3),name = 'image_input')

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=(16,8,3))(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides  =(2, 2), name='block1_pool')(x)

    # Block 2
    # x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    # x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # # Block 3
    # x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    # x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    # x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # Block 4
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(26, activation='softmax', name='predictions')(x)

    #Create your own model 
    my_model = Model(input=img_input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()

    return my_model


