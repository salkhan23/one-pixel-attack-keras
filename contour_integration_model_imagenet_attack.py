# ---------------------------------------------------------------------------------------
# Mobile Network Imagenet one pixle attack
# ---------------------------------------------------------------------------------------
from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

from keras import backend as keras_backend
from keras.preprocessing.image import load_img, img_to_array

from networks import alex_net_cont_int_model

from keras.applications.imagenet_utils import decode_predictions, preprocess_input

# Helper functions
import helper
from attack import PixelAttacker


def preprocessing_function(x):
    """
    Preprocessing function used during training

    :param x:
    :return:
    """
    # return imagenet_utils.preprocess_input(x, mode='tf')
    x = (x - x.min()) / (x.max() - x.min())
    return x


if __name__ == '__main__':
    np.random.seed(100)
    keras_backend.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Load MobileNet
    # -----------------------------------------------------------------------------------
    print("Loading model and weights ...")

    weights_file = "./networks/models/partially_trained_cont_int_model_weights.hf"

    model = alex_net_cont_int_model.build_full_contour_integration_model(
        rf_size=35,
        inner_leaky_relu_alpha=0.9,
        outer_leaky_relu_alpha=1.0,
        l1_reg_loss_weight=0.0005,
        weights_file=weights_file
    )

    do_not_train = ['conv_1', 'contour_integration_layer']

    for layer in model.layers:
        if layer.name in do_not_train:
            layer.trainable = False

    for layer in model.layers:
        print("print {} is trainable {}".format(layer.name, layer.trainable))

    # model.summary()

    # -----------------------------------------------------------------------------------
    # Handle ImageNet
    # -----------------------------------------------------------------------------------
    # Class index to class name converter(class name is the word description)
    with open('data/imagenet_classes.pkl', 'rb') as f:
        class_names = pickle.load(f)
    word_to_class = {w: i for i, w in enumerate(class_names)}

    with open('data/sample_images/data_key.pickle', 'rb') as handle:
        data_key = pickle.load(handle)

    fnames = sorted(data_key.keys())
    names = [data_key[fname][1] for fname in fnames]

    images = []
    labels = []

    for fname, name in zip(fnames, names):
        img = load_img(os.path.join("data/sample_images", fname), target_size=(227, 227))
        np_imgs = img_to_array(img, data_format='channels_first')
        images.append(np_imgs)
        labels.append([word_to_class[name]])

    images = np.array(images)
    labels = np.array(labels)

    print("Images shape {}, min {}, max {}".format(
        images.shape, np.min(images), np.max(images)))

    # -----------------------------------------------------------------------------------
    # Setup the Attacker
    # -----------------------------------------------------------------------------------
    models = [model]

    test_tuple = (images.copy(), labels)

    attacker = PixelAttacker(
        models,
        test_tuple,
        class_names,
        dimensions=(227, 227),
        preprocessing_cb=preprocessing_function
    )

    # -----------------------------------------------------------------------------------
    # Evaluate Model Accuracy
    # -----------------------------------------------------------------------------------
    correct_imgs = attacker.correct_imgs
    network_stats = attacker.network_stats

    correct_imgs = pd.DataFrame(
        correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
    network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    print(network_stats)
    print("Number of correctly classified images {}".format(len(correct_imgs)))

    # -----------------------------------------------------------------------------------
    # Full Attack
    # -----------------------------------------------------------------------------------
    print("Starting Full attack ...")

    start_time = datetime.now()

    untargeted_full_attack_results = attacker.attack_all(
        models,
        samples=300,
        targeted=False,
        pixels=[1],
        preprocessing_cb=preprocessing_function
    )

    print("Processing took {}".format(datetime.now() - start_time))

    # Load the results
    untargeted = helper.load_results()

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff',
               'prior_probs', 'predicted_probs', 'perturbation']

    untargeted_results = pd.DataFrame(untargeted, columns=columns)

    print(helper.attack_stats(untargeted_results, models, network_stats))
