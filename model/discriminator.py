from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

class Discriminator:
    def __init__(self, HRShape):
        self.HRShape = HRShape
    
    def Normalize(self, **kwargs):
        return Lambda(lambda x: x / 127.5 - 1, **kwargs)
    
    def CreateDiscriminatorBlock(self, model, filters, strides = 1, bn = True):
        x = Conv2D(filters = filters, kernel_size = 3, strides = strides, padding = "same")(model)
        x = LeakyReLU(alpha = 0.2)(x)
        if bn:
            x = BatchNormalization()(x)
        return x

    def BuildDiscriminator(self):
        model_in = Input(shape = self.HRShape)

        model = self.Normalize()(model_in)

        model = self.CreateDiscriminatorBlock(model, 64, bn = False)
        model = self.CreateDiscriminatorBlock(model, 64, strides = 2)

        model = self.CreateDiscriminatorBlock(model, 64 * 2)
        model = self.CreateDiscriminatorBlock(model, 64 * 2, strides = 2)

        model = self.CreateDiscriminatorBlock(model, 64 * 6)
        model = self.CreateDiscriminatorBlock(model, 64 * 6, strides = 2)

        model = self.CreateDiscriminatorBlock(model, 64 * 8)
        model = self.CreateDiscriminatorBlock(model, 64 * 8, strides = 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Dense(1, activation = "sigmoid")(model)
        
        return Model(inputs = model_in, outputs = model)
