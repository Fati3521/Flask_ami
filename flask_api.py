import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import flasgger
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.image import decode_png
from tensorflow.image import resize
from tensorflow.io import read_file
from flask import Flask, request
from glob import glob
from os import getcwd
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from flasgger import Swagger
from objects.WeightedCrossEntropy import WeightedCrossEntropy
from objects.BalancedCrossEntropy import BalancedCrossEntropy
from objects.TimeCallBack import TimingCallback
from werkzeug.utils import secure_filename

import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template

IMAGE_SIZE = 256


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims(image_tensor, axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def read_image(image_path, mask=False):
    image = read_file(image_path)
    if mask:
        image = decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])

    else:
        image = decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255

    return image


def plot_samples_matplotlib(display_list, filename, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.savefig(filename)  # 'static/media/inference.jpg')


def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        filename = image_file
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], filename, figsize=(18, 14)
        )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


metrics_wce = WeightedCrossEntropy
metrics_bce = BalancedCrossEntropy
cb = TimingCallback()
deeplabv3_plus = load_model('model/deeplabv3_plus', compile=False,
                            custom_objects={'WeightedCrossEntropy': metrics_wce,
                                            'BalancedCrossEntropy': metrics_bce,
                                            'TimingCallback': cb})

UPLOAD_FOLDER = 'static/media/prediction'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    try:
        filelist = glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
        for filename in filelist:
            os.remove(filename)
    except FileNotFoundError as e:
        print(e)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        colormap = loadmat(
            "city_colormap.mat"
        )["colormap"]
        colormap = colormap * 100
        colormap = colormap.astype(np.uint8)
        plot_predictions([os.path.join(app.config['UPLOAD_FOLDER'], filename)], colormap, model=deeplabv3_plus)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename='media/prediction/' + filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        #return render_template('upload.html')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='media/prediction/inference.jpg'), code=301)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
