from keras.layers import Input, Conv2D, BatchNormalization, \
    Activation, Dropout, Add, Conv2DTranspose, LeakyReLU, \
    Concatenate
from keras.initializers import RandomNormal, Zeros
from keras.utils import Sequence
from keras import Model
import numpy as np
import cv2
import os


class DataGenerator(Sequence):
    def __init__(self, ids, filepath, batch_size=16, image_size=(256, 256), channels=3):
        self.ids=ids
        self.filepath=filepath
        self.batch_size=batch_size
        self.image_size=image_size
        self.channels=channels
        
    def __len__(self):
        return (len(self.ids) // self.batch_size)
            
    def __getitem__(self, index):
        indexes = range(
            index * self.batch_size,
            index * self.batch_size + self.batch_size
        )
        
        ids_to_load = [self.ids[k] for k in indexes]
        X = self.__generate_x(ids_to_load)
        y = self.__generate_y(ids_to_load)
        l = []
        for i, m in enumerate(X):
            l.append(m.reshape(256, 256, 3))
        X = np.array(l)
        l = []
        for i, m in enumerate(y):
            l.append(m.reshape(256, 256, 3))
        y = np.array(l)
        return X, y
            
    def __generate_x(self, id_names):
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.channels))
        for i,id_n in enumerate(id_names):
            img = cv2.imread(os.path.join(self.filepath, 'drawing', id_n))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = img.astype(np.float32)/255.
            #img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

            img = np.array(img, np.float32)
            img *= (1.0/img.max())
            X[i,:,:,:] = img.reshape(self.image_size[0], self.image_size[1], self.channels)
        return X
        
    def __generate_y(self, id_names):
        y = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.channels))
        for i,id_n in enumerate(id_names):
            img = cv2.imread(os.path.join(self.filepath, 'original', id_n))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = img.astype(np.float32)/255.
            #img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
            
            img = np.array(img, np.float32)
            img *= (1.0/img.max())
            y[i,:,:,:] = img.reshape(self.image_size[0], self.image_size[1], self.channels)
        return y


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def residual_block(feature, dropout=False):
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return Add()([feature, x])

def conv_block(feature, out_channel, downsample=True, dropout=False):
    if downsample:
        x = Conv2D(out_channel, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    else:
        x = Conv2DTranspose(out_channel, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    return x

def get_generator(n_block=3):
    input = Input(shape=(image_size[0], image_size[1], input_channel))
    x = Conv2D(64, kernel_size=7, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(input)  # use reflection padding instead
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # downsample
    x = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # downsample
    x = Conv2D(256, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for i in range(n_block):
        x = residual_block(x)
    # upsample
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # upsample
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # out
    x = Conv2D(output_channel, kernel_size=7, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)  # use reflection padding instead
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    generator = Model(inputs=input, outputs=x)
    return generator

def get_generator_unet(n_block=3):
    input = Input(shape=(image_size[0], image_size[1], input_channel))
    # encoder
    e0 = Conv2D(
        64,
        kernel_size=4,
        padding='same',
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        bias_initializer=Zeros()
    )(input)  # use reflection padding instead

    e0 = BatchNormalization()(e0)
    e0 = Activation('relu')(e0)
    e1 = conv_block(e0, 128, downsample=True, dropout=False)  # 1/2
    e2 = conv_block(e1, 256, downsample=True, dropout=False)  # 1/4
    e3 = conv_block(e2, 512, downsample=True, dropout=False)  # 1/8
    e4 = conv_block(e3, 512, downsample=True, dropout=False)  # 1/16
    e5 = conv_block(e4, 512, downsample=True, dropout=False)  # 1/32
    e6 = conv_block(e5, 512, downsample=True, dropout=False)  # 1/64
    e7 = conv_block(e6, 512, downsample=True, dropout=False)  # 1/128
    # decoder
    d0 = conv_block(e7, 512, downsample=False, dropout=True)  # 1/64
    d1 = Concatenate(axis=-1)([d0, e6])
    d1 = conv_block(d1, 512, downsample=False, dropout=True)
    d2 = Concatenate(axis=-1)([d1, e5])
    d2 = conv_block(d2, 512, downsample=False, dropout=True)  # 1/16
    d3 = Concatenate(axis=-1)([d2, e4])
    d3 = conv_block(d3, 512, downsample=False, dropout=True)  # 1/8
    d4 = Concatenate(axis=-1)([d3, e3])
    d4 = conv_block(d4, 256, downsample=False, dropout=True)  # 1/4
    d5 = Concatenate(axis=-1)([d4, e2])
    d5 = conv_block(d5, 128, downsample=False, dropout=True)  # 1/2
    d6 = Concatenate(axis=-1)([d5, e1])
    d6 = conv_block(d6, 64, downsample=False, dropout=True)  # 1
    # out
    x = Conv2D(
        output_channel,
        kernel_size=3,
        padding='same',
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        bias_initializer=Zeros()
    )(d6)  # use reflection padding instead
    
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    generator = Model(inputs=input, outputs=x)
    return generator

def get_generator_training_model(generator, discriminator):
    imgA = Input(shape=(image_size[0], image_size[1], input_channel))
    imgB = Input(shape=(image_size[0], image_size[1], input_channel))
    fakeB = generator(imgA)
    # discriminator.trainable=False
    realA_fakeB = Concatenate()([imgA, fakeB])
    pred_fake = discriminator(realA_fakeB)
    generator_training_model = Model(inputs=[imgA, imgB], outputs=[pred_fake, fakeB])
    return generator_training_model

def get_discriminator(n_layers=4, use_sigmoid=True):
    input = Input(shape=(image_size[0], image_size[1], input_channel + output_channel))
    x = Conv2D(
        64,
        kernel_size=4,
        padding='same',
        strides=2,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        bias_initializer=Zeros()
    )(input)

    x = LeakyReLU(alpha=0.2)(x)
    for i in range(1, n_layers):
        x = Conv2D(64 * 2 ** i, kernel_size=4, padding='same', strides=2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(64 * 2 ** n_layers, kernel_size=4, padding='same', strides=1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, kernel_size=4, padding='same', strides=1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)

    if use_sigmoid:
        x = Activation('sigmoid')(x)
    discriminator = Model(inputs=input, outputs=x)
    return discriminator
