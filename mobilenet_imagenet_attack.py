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
from datetime import datetime

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
    model = keras.applications.MobileNet()

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

    # # -----------------------------------------------------------------------------------
    # # Model Accuracy
    # # -----------------------------------------------------------------------------------
    models = [model]

    network_stats, correct_imgs = helper.evaluate_models(models, processed_images, labels)

    correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
    network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])
    print(network_stats)

    # Set up the attacker
    models = [model]

    # The data set we will try to attack
    test_tuple = (processed_images, labels)

    attacker = PixelAttacker(models, test_tuple, class_names, dimensions=(224, 224))

    # -----------------------------------------------------------------------------------
    # Single Image Attack
    # -----------------------------------------------------------------------------------
    target_img_idx = 101

    start_time = datetime.now()
    result = attacker.attack(target_img_idx, model, maxiter=100, verbose=True)
    print("Processing took {}".format(datetime.now() - start_time))

    # Decode the Results
    print("Was Attack successful {}. Modified pixel {}".format(result[5], result[-1]))
    print("True Class ({}:{}), Final Predicted class ({}:{})".format(
        result[3], class_names[result[3]], result[4], class_names[result[4]]))
    print("Drop in confidence in true class {}".format(result[6]))

    # Show the difference between start and stop image
    changed_pixel = result[-1]
    attack_image = helper.perturb_image(changed_pixel, processed_images[target_img_idx, ])[0]

    true_class = labels[target_img_idx, 0]

    prior_confidence = model.predict(np.expand_dims(processed_images[target_img_idx,], axis=0))[0][true_class]
    post_confidence = model.predict(np.expand_dims(attack_image, axis=0))[0][true_class]

    success = attacker.attack_success(
        changed_pixel, processed_images[target_img_idx,], true_class, model, verbose=True)

    print('Prior confidence {}, After Attack confidence {}. Attack was successful {}'.format(
        prior_confidence, post_confidence, success))


    # plt.figure()
    # plt.imshow(attack_image)
    # plt.title("Attack Image Raw")
    #
    # plt.figure()
    # new_image = (attack_image - attack_image.min()) / (attack_image.max() - attack_image.min()) * 255.0
    # helper.plot_image(new_image)

    # -------------------------------------------------------------------------------------------------
    # Attack Evaluation
    # --------------------------------------------------------------------------------------------------
    print ("Starting Full attack ...")

    # The full attack
    start_time = datetime.now()
    untargeted = attacker.attack_all(models, samples=50, targeted=False)
    print("Processing took {}".format(datetime.now() - start_time))

    # Load the results
    untargeted = helper.load_results()

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation']

    untargeted_results = pd.DataFrame(untargeted, columns=columns)

    helper.attack_stats(untargeted_results, models, network_stats)
