# ---------------------------------------------------------------------------------------
# Mobile Network Imagenet one pixle attack
# ---------------------------------------------------------------------------------------
from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime

import keras
from keras.preprocessing.image import load_img, img_to_array

# from keras.applications.imagenet_utils import decode_predictions, preprocess_input
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
    print("Loading the Model...")
    model = keras.applications.MobileNet()

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
        img = load_img(os.path.join("data/sample_images", fname), target_size=(224, 224))
        x = img_to_array(img, data_format='channels_last')
        images.append(x)
        labels.append([word_to_class[name]])

    images = np.array(images)
    labels = np.array(labels)

    # Normalize ImageNet Images
    # processed_images = preprocess_input(images.copy(), data_format='channels_last')
    # It is important to use the correct preprocessing function for the model
    # processed_images = mobilenet_preprocess_input(images.copy())
    processed_images = images.copy()

    print("Processed Images shape {}, min {}, max {}".format(
        processed_images.shape, np.min(processed_images), np.max(processed_images)))

    # # -----------------------------------------------------------------------------------
    # # Model Accuracy
    # # -----------------------------------------------------------------------------------
    print("Evaluating Model Accuracy - Regular Images")
    models = [model]

    network_stats, correct_imgs = helper.evaluate_models(
        models,
        processed_images,
        labels,
        preprocessing_cb=mobilenet_preprocess_input
    )

    correct_imgs = pd.DataFrame(
        correct_imgs,
        columns=['name', 'img', 'label', 'confidence', 'pred']
    )

    network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])
    print(network_stats)

    print("Number of correctly Classifiered images {}".format(len(correct_imgs)))
    # -----------------------------------------------------------------------------------
    #  Adverserial Attacks
    # -----------------------------------------------------------------------------------
    # Set up the attacker
    models = [model]

    test_tuple = (processed_images.copy(), labels)
    attacker = PixelAttacker(models, test_tuple, class_names, dimensions=(224, 224),
                             preprocessing_cb=mobilenet_preprocess_input)

    # -----------------------------------------------------------------------------------
    # Original Image
    # -----------------------------------------------------------------------------------
    print("Starting Attack {}".format('*' * 80))
    target_img_idx = 311

    # First Verify Model predicts the correct label80
    print("For Image @ index {}, True label {}".format(target_img_idx, names[target_img_idx]))

    preprocessed_tgt_image = mobilenet_preprocess_input(processed_images[target_img_idx,].copy())
    prediction = model.predict(np.expand_dims(preprocessed_tgt_image, axis=0))
    # helper.plot_image(images[target_img_idx,])

    print("Model predictions")
    predictions = mobilenet_decode_predictions(prediction)[0]
    for item in predictions:
        print(item)

    # Change one pixel
    changed_pixel = np.array([200,100, 255,0, 255])
    attack_image = helper.perturb_image(
        changed_pixel,
        processed_images[target_img_idx, ].copy())[0]

    preprocessed_attack_image = mobilenet_preprocess_input(attack_image.copy())
    prediction = model.predict(np.expand_dims(preprocessed_attack_image, axis=0))

    print("Attacked Image prediction")
    predictions = mobilenet_decode_predictions(prediction)[0]
    for item in predictions:
        print(item)

    # -----------------------------------------------------------------------------------
    # Single Image Attack
    # -----------------------------------------------------------------------------------
    print("Single Image Attack ...")
    start_time = datetime.now()
    result = attacker.attack(
        target_img_idx,
        model,
        maxiter=5,
        verbose=True,
        preprocessing_cb=mobilenet_preprocess_input
    )

    print("Processing took {}".format(datetime.now() - start_time))

    # Decode the Results
    print("Was attack successful {}. Modified pixel {}".format(result[5], result[-1]))
    print("True Class ({}:{}), Final Predicted class ({}:{})".format(
        result[3], class_names[result[3]], result[4], class_names[result[4]]))
    print("Drop in confidence in true class {}".format(result[6]))

    # Show the difference between start and stop image
    changed_pixel = result[-1]
    attack_image = helper.perturb_image(
        changed_pixel,
        processed_images[target_img_idx, ].copy())[0]

    true_class = labels[target_img_idx, 0]

    preprocessed_original_image = mobilenet_preprocess_input(processed_images[target_img_idx,].copy())
    preprocessed__attack_image = mobilenet_preprocess_input(attack_image.copy())

    prior_confidence = model.predict(
        np.expand_dims(preprocessed_original_image, axis=0))[0][true_class]
    post_confidence = model.predict(
        np.expand_dims(preprocessed__attack_image, axis=0))[0][true_class]

    success = attacker.attack_success(
        changed_pixel, processed_images[target_img_idx, ].copy(), true_class, model, verbose=True)

    print('Prior confidence {}, After Attack confidence {}. Attack was successful {}'.format(
        prior_confidence, post_confidence, success))


   # plt.figure()
   # plt.imshow(attack_image)
   # plt.title("Attack Image Raw")

    #plt.figure()
    new_image = (attack_image - attack_image.min()) / (attack_image.max() - attack_image.min()) * 255.0
    #helper.plot_image(new_image)

    # -------------------------------------------------------------------------------------------------
    # Attack Evaluation
    # --------------------------------------------------------------------------------------------------
    print ("Starting Full attack ...")

    # The full attack
    start_time = datetime.now()
    untargeted = attacker.attack_all(models, samples=300, targeted=False, pixels=[1],preprocessing_cb=mobilenet_preprocess_input )
    print("Processing took {}".format(datetime.now() - start_time))

    # Load the results
    untargeted = helper.load_results()

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation']

    untargeted_results = pd.DataFrame(untargeted, columns=columns)

    helper.attack_stats(untargeted_results, models, network_stats)
