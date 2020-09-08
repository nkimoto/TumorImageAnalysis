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
import glob
import matplotlib.pyplot as plt
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-t",
        "--threshold",
        help="The threshold of immunostaining result (default: 90)",
        type=float,
        default=90,
    )
    parser.add_argument(
        "-s",
        "--stain",
        help="The type of immunostaining (default: ER)",
        type=str,
        choices=["ER", "PgR", "Ki-67"],
        default="ER",
    )
    args = parser.parse_args()
    return args


def preprocess(input_pkl):
    # Load image dict
    with open(input_pkl, "rb") as rf:
        im_dict = pickle.load(rf)

    # Create case, control list
    case = []
    for i in im_dict["case"].values():
        case.extend(i)
    len(case)

    control = []
    for i in im_dict["control"].values():
        control.extend(i)
    len(control)

    # Create data_x, data_y
    data_x = []
    data_y = []
    data_x.extend(case)
    data_y.extend([1] * len(case))
    data_x.extend(control)
    data_y.extend([0] * len(control))
    assert len(data_x) == len(data_y)

    # np.arrayに変換
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    # 学習用データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # 正規化
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Check train image size
    print("X_train: ")
    print(x_train.shape)
    print("X_test: ")
    print(x_test.shape)
    print("y_train: ")
    print(y_train.shape)
    print("y_test: ")
    print(y_test.shape)
    return x_train, x_test, y_train, y_test


def plot_results(history, stain):
    # Accuracy
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    name = f"val_acc_{stain}.jpg"
    plt.savefig(name, bbox_inches="tight")

    # loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    name = f"val_loss_{stain}.jpg"
    plt.savefig(name, bbox_inches="tight")


def main(stain="ER"):
    x_train, x_test, y_train, y_test = preprocess(input_pkl=f"im_dict_{stain}.pickle")
    # CNN
    model.add(
        Conv2D(
            64, (3, 3), activation="relu", padding="same", input_shape=x_train.shape[1:]
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )

    plot_model(
        model, to_file="cnn_model_{stain}.png", show_shapes=True, show_layer_names=True
    )

    filepath = f"cnn_model_{stain}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )

    # Optimize model
    epochs = 20
    history = model.fit(
        x_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[checkpoint]
    )
    score = model.evaluate(x_test, y_test, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    plot_results()
