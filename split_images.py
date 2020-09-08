#! coding: utf-8

import sys
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from logzero import logger
import pickle
import argparse


REF_FILE = "AI_data20200812.xlsx"
LABEL_DIR_NAME = "0.05x_seg_to_0.5x"
SAMPLE_DIR_NAME = "HE_10x_exported_0.5x"
Image.MAX_IMAGE_PIXELS = 10000000000


def label_sample_generator(ref, stain="ER", threshold=90):
    for l, s in zip(
        sorted(os.listdir(LABEL_DIR_NAME)), sorted(os.listdir(SAMPLE_DIR_NAME))
    ):
        sample_name = l.split(" - ")[0].strip()
        if sample_name in ref["検体番号"].to_list():
            percent = ref.loc[ref["検体番号"] == sample_name, stain].iloc[0]
            group = 0
            if not isinstance(percent, (int, float)):
                continue
            elif percent > threshold:
                group = 1
            #             elif percent == 0:
            #                 group = 0
            else:
                group = 0
            yield l, s, group


def split_img(img, px=224):
    """Split Image Object to specified px"""
    w_list = [(px * i, px * (i + 1)) for i in range(img.shape[0] // px)]
    h_list = [(px * i, px * (i + 1)) for i in range(img.shape[1] // px)]
    print(f"Dimention : {len(w_list)}, {len(h_list)}")
    return [img[w_s:w_e, h_s:h_e, :] for w_s, w_e in w_list for h_s, h_e in h_list]


def get_stained_segment_ids(sub_im_list, threshold=200, percent=80, px=224):
    ids = []
    for id_, sub_im in enumerate(sub_im_list):
        r, g, b = cv2.split(sub_im)
        if np.count_nonzero(r > threshold) > px ** 2 * percent / 100:
            ids.append(id_)
    return ids


def get_stained_im_list(sub_im_list, ids, verbose=False):
    target_im_list = []

    for id_ in ids:
        sub_im = sub_im_list[id_]
        target_im_list.append(sub_im)
        if verbose:
            print(f"id: {id_}")
            plt.imshow(sub_im)
            plt.show()
    return target_im_list


def convert_PIL_to_array(img):
    return np.asarray(img)


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
        help="The type of immunostaining (required)",
        type=str,
        choices=["ER", "PgR", "Ki-67"],
        required=True,
    )
    args = parser.parse_args()
    return args


def main(stain, threshold):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    ref = pd.read_excel(REF_FILE)
    gen = label_sample_generator(ref, stain=stain, threshold=threshold)
    control, case = {}, {}
    im_dict = {"case": {}, "control": {}}
    logger.info("Start Processing!")
    for l, s, g in gen:
        sample_name = l.split(" - ")[0].strip()
        l_im = Image.open(os.path.join(LABEL_DIR_NAME, l))
        s_im = Image.open(os.path.join(SAMPLE_DIR_NAME, s))
        # 画像をarrayに変換
        l_im_list = convert_PIL_to_array(l_im)
        s_im_list = convert_PIL_to_array(s_im)
        # 貼り付け
        logger.info(f"Handling {l}, {s} ...")
        l_sub_im_list = split_img(l_im_list)
        s_sub_im_list = split_img(s_im_list)
        assert all([image.shape == (224, 224, 3) for image in s_sub_im_list])
        target_ids = get_stained_segment_ids(l_sub_im_list)
        target_im_list = get_stained_im_list(s_sub_im_list, target_ids)
        if g:
            im_dict["case"][sample_name] = target_im_list
        else:
            im_dict["control"][sample_name] = target_im_list
    with open(f"im_dict_{stain}_{threshold}.pickle", "wb") as wf:
        pickle.dump(im_dict, wf)


if __name__ == "__main__":
    args = parse_args()
    main(args.stain, args.threshold)
