# -------------------------------------------------------------------------------------------------
#  New Model of Contour Integration.
#
#  Compared to  previous models, the contour integration kernel used is 3D and connects across
#  feature maps
#
# Author: Salman Khan
# Date  : 27/04/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.engine.topology import Layer
import keras.activations as activations
from keras.regularizers import l1, Regularizer
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, \
    Flatten, Dense, Dropout, Activation
from keras.models import Model

from networks import alex_net

reload(alex_net)

class ContourGainCalculatorLayer(Layer):
    def __init__(self, tgt_filt_idx, **kwargs):
        """
        A layer that calculates the enhancement gain of the neuron focused on the center of the image
        and at channel index = tgt_filt_idx.

        TODO: Make the tgt_filt_idx configurable. So that it can be specified in the call function.
        TODO: This will allow all contour integration kernels to be trained in the same layer.
        TODO: Alternatively try lambda layers

        :param tgt_filt_idx:
        :param kwargs:
        """
        self.tgt_filt_idx = K.variable(tgt_filt_idx, dtype='int32')
        super(ContourGainCalculatorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContourGainCalculatorLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], 1

    def call(self, inputs, **kwargs):
        """
        Calculate the enhancement gain of the neuron at the center of the image.

        :param inputs: a list/tuple of (feature extract activation, contour enhanced activation)
        :param kwargs:

        :return:
        """
        feature_act = inputs[0]
        contour_act = inputs[1]

        _, ch, r, c = K.int_shape(inputs[0])
        center_neuron_loc = (r >> 1, c >> 1)

        gain = contour_act[:, self.tgt_filt_idx, center_neuron_loc[0], center_neuron_loc[1]] / \
            (feature_act[:, self.tgt_filt_idx, center_neuron_loc[0], center_neuron_loc[1]] + 1e-4)

        return K.expand_dims(gain, axis=-1)


class FeatureMapL1Regularizer(Regularizer):

    def __init__(self, tgt_filt_idx, l1=0.):
        self.tgt_filt_idx = tgt_filt_idx
        self.l1 = K.cast_to_floatx(l1)

    def __call__(self, x):
        regularization = K.sum(K.abs(x[:, :, :, self.tgt_filt_idx]) * self.l1)

        return regularization

    def get_config(self):
        return {'tgt_filt_idx': int(self.tgt_filt_idx), 'l1': float(self.l1)}


class ContourIntegrationLayer3D(Layer):

    def __init__(self, tgt_filt_idx, inner_leaky_relu_alpha, outer_leaky_relu_alpha, rf_size=25,
                 activation=None, l1_reg_loss_weight=0.001, **kwargs):
        """
        Contour Integration layer. Different from previous contour integration layers,
        the contour integration kernel is 3D and allows connections between feature maps

        :param rf_size:
        :param activation:
        :param kwargs:
        """

        if 0 == (rf_size & 1):
            raise Exception("Specified RF size should be odd")

        self.tgt_filt_idx = K.variable(tgt_filt_idx, dtype='int32')
        self.n = rf_size
        self.activation = activations.get(activation)
        self.inner_leaky_relu_alpha = inner_leaky_relu_alpha
        self.outer_leaky_relu_alpha = outer_leaky_relu_alpha
        self.l1_reg_loss_weight = l1_reg_loss_weight
        super(ContourIntegrationLayer3D, self).__init__(**kwargs)

    def build(self, input_shape):
        _, ch, r, c = input_shape
        # print("Build Fcn: Channel First Input shape ", input_shape)

        # Todo: Check which dimension is input and which one is output
        self.kernel = self.add_weight(
            shape=(self.n, self.n, ch, ch),
            initializer='glorot_normal',
            name='kernel',
            trainable=True,
            regularizer=FeatureMapL1Regularizer(self.tgt_filt_idx, self.l1_reg_loss_weight)
        )

        self.bias = self.add_weight(
            shape=(ch,),
            initializer='zeros',
            name='bias',
            trainable=True,
            # regularizer=l1(0.05)
        )

        super(ContourIntegrationLayer3D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):
        """
        Selectively enhance the gain of neurons in the feature extracting activation volume that
        are part of a smooth contour.

        :param inputs:
        :param kwargs:
        :return:
        """
        _, ch, r, c = K.int_shape(inputs)
        # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

        outputs = K.conv2d(inputs, self.kernel, strides=(1, 1), padding='same')

        outputs = K.bias_add(outputs, self.bias)
        outputs = outputs * inputs

        # outputs = self.activation(outputs) + inputs
        outputs = K.relu(outputs, alpha=self.inner_leaky_relu_alpha) + inputs

        outputs = K.relu(outputs, alpha=self.outer_leaky_relu_alpha)

        return outputs


def build_contour_integration_model(
        tgt_filt_idx, rf_size=25, inner_leaky_relu_alpha=0.7, outer_leaky_relu_alpha=0.7, l1_reg_loss_weight=0.01):
    """
    Build a (short) model of 3D contour integration that can be used to train the model.

    THis is build on after the first feature extracting layer of object classifying network,
    and only trains the contour integration layer. THe complete model can still be used for
    object classification

    :param l1_reg_loss_weight:
    :param outer_leaky_relu_alpha:
    :param inner_leaky_relu_alpha:
    :param rf_size:
    :param tgt_filt_idx:
    :return:
    """
    input_layer = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(input_layer)

    contour_integrate_layer = ContourIntegrationLayer3D(
        tgt_filt_idx=tgt_filt_idx,
        rf_size=rf_size,
        inner_leaky_relu_alpha=inner_leaky_relu_alpha,
        outer_leaky_relu_alpha=outer_leaky_relu_alpha,
        l1_reg_loss_weight=l1_reg_loss_weight,
        name='contour_integration_layer')(conv_1)

    contour_gain_layer = ContourGainCalculatorLayer(
        tgt_filt_idx,
        name='gain_calculating_layer')([conv_1, contour_integrate_layer])

    model = Model(input_layer, outputs=contour_gain_layer)

    model.layers[1].trainable = False  # Set the feature extracting layer as untrainable.

    model.load_weights("trained_models/AlexNet/alexnet_weights.h5", by_name=True)
    model.compile(optimizer='Adam', loss='mse')

    return model


def build_full_contour_integration_model(
        weights_file=None, rf_size=35, inner_leaky_relu_alpha=0.9, outer_leaky_relu_alpha=1.,
        l1_reg_loss_weight=0.0005):
    """

    Build the full contour integration Alexnet Model
    Note:[1] Model needs to be complied fore use.
         [2] The name of the layers after the contour integration layer are changed from alexnet
             so weights of alexnet can be loaded safely.

    :param weights_file:
    :param rf_size:
    :param inner_leaky_relu_alpha:
    :param outer_leaky_relu_alpha:
    :param l1_reg_loss_weight:
    :return:
    """
    input_layer = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(input_layer)

    contour_integrate_layer = ContourIntegrationLayer3D(
        tgt_filt_idx=0,  # not important for full model
        rf_size=rf_size,
        inner_leaky_relu_alpha=inner_leaky_relu_alpha,
        outer_leaky_relu_alpha=outer_leaky_relu_alpha,
        l1_reg_loss_weight=l1_reg_loss_weight,
        name='contour_integration_layer')(conv_1)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(contour_integrate_layer)
    conv_2 = alex_net.crosschannelnormalization(name='Contrast_Normalization')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_22_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_22_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_22')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = alex_net.crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_33')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_44_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_44_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_44')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_55_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_55_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_5))
    conv_5 = Concatenate(axis=1, name='conv_55')([conv_5_1, conv_5_2])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_11')(dense_1)

    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_22')(dense_2)

    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_33')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(inputs=input_layer, outputs=prediction)

    if weights_file:
        model.load_weights(weights_file)

    return model


def update_contour_integration_kernel(model, new_tgt_filt_idx):
    """

    Given a contour integration model training model, update the contour integration kernel to be upgraded

    :param model:
    :param new_tgt_filt_idx:
    :return:
    """

    K.set_value(model.layers[2].tgt_filt_idx, new_tgt_filt_idx)
    K.set_value(model.layers[3].tgt_filt_idx, new_tgt_filt_idx)

    # remember to recompile the model
    return model


if __name__ == '__main__':

    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    tgt_kernel_idx = 5

    np.random.seed(7)  # Set the random seed for reproducibility

    # -----------------------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------------------
    print("Building the contour integration model...")
    cont_int_model = build_contour_integration_model(tgt_kernel_idx)
    # print cont_int_model.summary()

    # -----------------------------------------------------------------------------------
    # Validate the model is working properly
    # -----------------------------------------------------------------------------------
    image_name = "./data/sample_images/cat.7.jpg"

    # Option 1: Keras way
    # --------------------
    image = keras.preprocessing.image.load_img(
        image_name,
        target_size=[227, 227, 3]
    )

    # Takes care of putting channel first.
    input_image = keras.preprocessing.image.img_to_array(image)

    # # Option 2: pyplot and numpy only
    # # -------------------------------
    # # Note: This method only works for images that do not need to be resized.
    # image = plt.imread(image_name)
    # input_image = np.transpose(image, axes=(2, 0, 1))

    # plt.figure()
    # plt.imshow(image)

    y_hat = cont_int_model.predict(np.expand_dims(input_image, axis=0), batch_size=1)
    print("Model Prediction Enhancement Gain of {}".format(y_hat))
