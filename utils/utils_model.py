from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.losses import mean_squared_error
from keras.models import Model

def VGG(output_layer):
    model = VGG19(input_shape = (None, None, 3), include_top = False)
    model = Model(model.input, model.layers[output_layer].output)
    model.trainable = False

    return model

def ContentLoss(HR, SR):
    _VGG = VGG(20)
    SR = preprocess_input(SR)
    HR = preprocess_input(HR)
    SR_features = _VGG(SR)
    HR_features = _VGG(HR)

    return mean_squared_error(HR_features, SR_features)