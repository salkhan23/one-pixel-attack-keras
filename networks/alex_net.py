# -------------------------------------------------------------------------------------------------
#  This is the Alex Net model (original two separate streams model) trained on Imagenet.
#
#  Code Ref: https://github.com/heuritech/convnets-keras
#  The code has been updated to use Keras V2 APIs
#
#  NOTES:
#  [1] The model loads pre-trained weights that must be stored in trained_models/AlexNet/alexnet_weights.h5
#  [2] Weights can be found @ https://github.com/heuritech/convnets-keras
#  [3] A test image is also needed.
#
# Author: Salman Khan
# Date  : 21/07/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Activation, MaxPooling2D
from keras.layers import Dense, Flatten, ZeroPadding2D, Dropout
from keras.layers.core import Lambda
from keras.preprocessing.image import img_to_array, load_img
import keras.backend as K

from keras.layers import Conv2D, Concatenate


def get_target_feature_extracting_kernel(tgt_filt_idx):
    """
    Return the target kernel @ the specified index from the first feature extracting
    covolutional layer of alex_net

    :param tgt_filt_idx:
    :return:  target Feature extracting kernel
    """
    model = alex_net("trained_models/AlexNet/alexnet_weights.h5")
    feat_extract_kernels = K.eval(model.layers[1].weights[0])

    return feat_extract_kernels[:, :, :, tgt_filt_idx]


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original Alexnet
    """

    def f(x_in):
        b, ch, r, c = K.int_shape(x_in)
        # print("Cross Channel Normalization: Input shape [b,ch,r,c]", b, ch, r, c)

        half = n // 2
        square = K.square(x_in)
        extra_channels = K.spatial_2d_padding(
            K.permute_dimensions(square, (0, 2, 3, 1)), ((0, 0), (half, half)))

        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))

        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return x_in / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(x_in):
        div = K.int_shape(x_in)[axis] // ratio_split

        if axis == 0:
            output = x_in[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = x_in[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = x_in[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = x_in[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def alex_net(weights_path):
    """
    Note: Layer names have to stay the same, to enable loading pre-trained weights

    :param weights_path:
    :return: alexnet model
    """

    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='Contrast_Normalization')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_22_1')(splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_22_2')(splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_2')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_33')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_44_1')(splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_44_2')(splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_4')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_55_1')(splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_55_2')(splittensor(ratio_split=2, id_split=1)(conv_5))
    conv_5 = Concatenate(axis=1, name='conv_5')([conv_5_1, conv_5_2])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_11')(dense_1)

    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_22')(dense_2)

    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_33')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(inputs=inputs, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model


if __name__ == "__main__":

    plt.ion()

    import utils
    reload(utils)

    # 1. Load the model
    # --------------------------------------------------------------------
    # Model was originally defined with Theano backend.
    K.set_image_dim_ordering('th')
    alex_net_model = alex_net("trained_models/AlexNet/alexnet_weights.h5")
    alex_net_model.summary()

    # 2. Display First Layer Filters
    # --------------------------------------------------------------------
    weights_ch_last = alex_net_model.layers[1].weights[0]
    utils.display_filters(weights_ch_last)

    # 3. Display the activations of a test image
    # ---------------------------------------------------------------------
    # img = load_img("trained_models/data/sample_images/cat.7.jpg", target_size=(227, 227))
    img = load_img("trained_models/data/sample_images/zahra.jpg", target_size=(227, 227))
    plt.figure()
    plt.imshow(img)

    x = img_to_array(img)
    x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])

    # y_hat = model.predict(x, batch_size=1, verbose=1)
    # print("Prediction %s" % np.argmax(y_hat))

    utils.display_layer_activations(alex_net_model, 1, x)
