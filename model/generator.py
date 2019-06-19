from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input, add, Lambda
from keras.models import Model
from keras.layers.advanced_activations import PReLU
import tensorflow as tf

class Generator:
    def __init__(self, nBaseBlocks = 6, nResidualBlocks = 3, ScaleFactor = 2):
        self.nBaseBlocks = nBaseBlocks
        self.nResidualBlocks = nResidualBlocks
        self.ScaleFactor = ScaleFactor

    # Residual Block
    def AddResidualBlock(self, model, filters):
        x = Conv2D(filters = filters, kernel_size = 3, padding = "same")(model)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes = [1, 2])(x)
        x = Conv2D(filters = filters, kernel_size = 3, padding = "same")(x)
        x = BatchNormalization()(x)
            
        return add([model, x])

    def BuildRIRBlock(self, model):
        x = model
        for _ in range(self.nResidualBlocks):
            x = self.AddResidualBlock(x, 64)
        
        return add([model, x])

    def SubpixelConv2D(self, scale, **kwargs):
        return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)     

    # UpSampling Block
    def AddUpSamplingBlock(self, model, filters):
        x = Conv2D(filters = filters, kernel_size = 3, padding = "same")(model)
        x = self.SubpixelConv2D(2)(x)
        return PReLU(shared_axes = [1, 2])(x)

    def Normalization(self, **kwargs):
        return Lambda(lambda x: x / 127.5 - 1, **kwargs)
    
    def Denormalization(self, **kwargs):
        return Lambda(lambda x: (x + 1) * 127.5, **kwargs)

    def BuildGenerator(self):
        x_in = Input(shape = (None, None, 3))

        x = self.Normalization()(x_in)
        x = Conv2D(filters = 64, kernel_size = 9, padding = "same")(x)
        x = PReLU(shared_axes = [1, 2])(x)

        # Skip connection before Residual Blocks
        init_x = x

        # Adding RIR Blocks of 3 RBlocks in Each
        for _ in range(self.nBaseBlocks):
            x = self.BuildRIRBlock(x)
        
        x = Conv2D(filters = 64, kernel_size = 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = add([init_x, x])
        
        # Adding Up Sampling Blocks
        for _ in range(self.ScaleFactor // 2):
            x = self.AddUpSamplingBlock(x, 256)
        
        x = Conv2D(filters = 3, kernel_size = 9, padding = "same", activation='tanh')(x)
        x = self.Denormalization()(x)
        
        return Model(inputs = x_in, outputs = x)

def LoadGeneratorWithWeights(WeightsPath, ScalingFactor):
    Gen = Generator(3, 6, ScalingFactor).BuildGenerator()
    Gen.load_weights(WeightsPath)
    return Gen