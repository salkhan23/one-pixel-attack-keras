# ---------------------------------------------------------------------------------------
# Mobile Network Imagenet one pixle attack
# ---------------------------------------------------------------------------------------
from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import keras
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

from keras.applications.mobilenet import decode_predictions as mobilenet_decode_predictions
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input

# Helper functions
import helper
from attack import PixelAttacker


if __name__ == '__main__':
    plt.ion()
    np.random.seed(100)

    # -----------------------------------------------------------------------------------
    # Load MobileNet
    # -----------------------------------------------------------------------------------
    model = keras.applications.MobileNetV2()

    # -----------------------------------------------------------------------------------
    # Handle ImageNet
    # -----------------------------------------------------------------------------------
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
        img = load_img(os.path.join("data/sample_images", fname), target_size=(224, 224))
        x = img_to_array(img, data_format='channels_last')
        images.append(x)
        labels.append([word_to_class[name]])

    images = np.array(images)
    labels = np.array(labels)

    # Normalize ImageNet Images
    # processed_images = preprocess_input(images.copy(), data_format='channels_last')

    processed_images = mobilenet_preprocess_input(images.copy())
    print("It is import to use the correct preprocessing function for the model")

    print("Processed Images shape {}, min {}, max {}".format(
        processed_images.shape, np.min(processed_images), np.max(processed_images)))

    # -----------------------------------------------------------------------------------
    # Model Accuracy
    # -----------------------------------------------------------------------------------
    models = [model]

    network_stats, correct_imgs = helper.evaluate_models(models, processed_images, labels)

    correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
    network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    network_stats




