import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import glob
import matplotlib.pyplot as plt
import time
import os
import argparse

Image.MAX_IMAGE_PIXELS = 10000000000

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--img_file",
        help="Target image file (required)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--stain",
        help="The type of immunostaining (required)",
        type=str,
        choices=["ER", "PgR", "Ki-67"],
        required=True,
    )
    args = parser.parse_args()
    return args


def split_img(img, px=224):
    """Split Image Object to specified px"""
    w_list = [(px * i, px * (i + 1)) for i in range(img.shape[0] // px)]
    h_list = [(px * i, px * (i + 1)) for i in range(img.shape[1] // px)]
    print(f"Dimention : {len(w_list)}, {len(h_list)}")
    return [img[w_s:w_e, h_s:h_e, :] for w_s, w_e in w_list for h_s, h_e in h_list]

def convert_PIL_to_array(img):
    return np.asarray(img)

def preprocess(img_file):
    im = Image.open(img_file)
    im_list = convert_PIL_to_array(im)
    sub_im_list = split_img(im_list)

    data = np.array(sub_im_list).astype("float32")
    data = data / 255.0

    return data


def main(stain, img_file):
    data = preprocess(img_file)
    # CNN
    filepath = f"cnn_model_{stain}.hdf5"
    model = load_model(filepath)

    # Preduct
    result = model.predict(data)
    print(f"Predicted as 1: {(result == 1).sum()}")
    print(f"Predicted as 0: {(result == 0).sum()}")


if __name__ == "__main__":
    args = parse_args()
    main(args.stain, args.img_file)
