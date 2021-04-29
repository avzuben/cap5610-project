import os

import math
import numpy as np
from PIL import Image
from skimage.morphology import label

from pathlib import Path

from model import get_unet_3d
from data import DataGenerator

DATA_DIR = './ds/'
BATCH_SIZE = 10
SLICES = 10
HEIGHT = 352
WIDTH = 352
HEIGHT_UNET = 144
WIDTH_UNET = 144
N_CHANNELS = 1
N_CLASSES = 4
EPOCHS = 300

DIR_SEG = '../results/output-seg/'

# Create model
model = get_unet_3d(
    input_shape=(SLICES, HEIGHT_UNET, WIDTH_UNET, N_CHANNELS),
    n_class=N_CLASSES
)
model.summary()


def preprocess_input(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.concatenate((img, img, img), -1)
    return img


def mask_to_image(mask):
    img = np.zeros(mask.shape[0:2] + (3,))
    # First class
    idx = np.argmax(mask, axis=-1) == 1
    img[idx] = [1, 0, 0]
    # Second class
    idx = np.argmax(mask, axis=-1) == 2
    img[idx] = [0, 1, 0]
    # Third class
    idx = np.argmax(mask, axis=-1) == 3
    img[idx] = [0, 0, 1]
    return img


def model_folder(pat):
    folder = pat % 5
    if folder == 0:
        folder = 5
    return folder


def postprocess_prediction(seg):
    rv = get_largest_region(seg, 1)
    myo = get_largest_region(seg, 2)
    lv = get_largest_region(seg, 3)
    return rv + myo + lv


def get_largest_region(seg, class_label):
    # basically look for connected components and choose the largest one, delete everything else
    new_seg = np.zeros(seg.shape)
    for i in range(seg.shape[0]):
        mask = seg[i] == class_label
        if np.sum(seg[i, mask]) > 0:
            lbls = label(mask, 8)
            lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
            largest_region = np.argmax(lbls_sizes[1:]) + 1
            new_seg[i, lbls == largest_region] = seg[i, lbls == largest_region]
    return new_seg


def get_center_point(mask):
    no_mask = False
    if len(mask[mask != 0]):
        yp, xp = np.where(mask != 0)
        x_min = np.min(xp)
        x_max = np.max(xp)
        y_min = np.min(yp)
        y_max = np.max(yp)
    else:
        x_min = 0
        x_max = mask.shape[1] - 1
        y_min = 0
        y_max = mask.shape[0] - 1
        no_mask = True

    return (x_min + x_max) / 2, (y_min + y_max) / 2, no_mask


def crop_images(data, mask):
    x = list()
    y = list()

    for m in mask:
        xc, yc, no_mask = get_center_point(m)
        if not no_mask:
            x.append(xc)
            y.append(yc)

    if len(x) == 0:
        x.append(WIDTH // 2)
        y.append(HEIGHT // 2)

    xc = np.round(np.mean(x)).astype(np.int)
    yc = np.round(np.mean(y)).astype(np.int)

    return data[:, yc - HEIGHT_UNET//2:yc + HEIGHT_UNET//2, xc - WIDTH_UNET//2:xc + WIDTH_UNET//2, :], mask[:, yc - HEIGHT_UNET//2:yc + HEIGHT_UNET//2, xc - WIDTH_UNET//2:xc + WIDTH_UNET//2], xc, yc


val_patients = np.concatenate([np.arange(1, 101, 5), np.arange(2, 101, 5), np.arange(3, 101, 5), np.arange(4, 101, 5), np.arange(5, 101, 5)], 0)

for pat in val_patients:

    model_n = model_folder(pat)
    VAL_SET = str(model_n)

    weight_path = './output-3dn/' + VAL_SET + '/best-weights'

    model.load_weights(weight_path)

    pred_seg = np.load(DIR_SEG + VAL_SET + '/' + str(pat).zfill(3) + '.npy', allow_pickle=True)

    ed_idx = np.arange(0, len(pred_seg), 2)
    es_idx = np.arange(1, len(pred_seg), 2)

    pred_seg = np.concatenate(
        [[postprocess_prediction(np.round(pred_seg[ed_idx]))], [postprocess_prediction(np.round(pred_seg[es_idx]))]],
        axis=0)
    pred_seg = pred_seg.reshape(pred_seg.shape[:-1])


    def save_image(img, pat, name):
        img *= 255
        im = Image.fromarray(img.astype(np.uint8))
        im.convert('RGB').save('./pred-3dn/' + VAL_SET + '/' + pat + '/' + name + '.jpg', 'JPEG')


    Path('./pred-3dn/' + VAL_SET + '/' + str(pat).zfill(3)).mkdir(parents=True, exist_ok=True)
    generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=1,
                              height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                              custom_keys=[pat], shuffle=False, to_fit=True)

    pred_multi_input_heart_locator = None

    print('Patient ' + str(pat))

    X = np.zeros((2, generator.__len__() // 2, HEIGHT, WIDTH, N_CHANNELS))
    y = np.zeros((2, generator.__len__() // 2, HEIGHT, WIDTH))

    for i in range(generator.__len__()):
        X_slice, y_slice = generator.__getitem__(i)
        y_slice = np.argmax(y_slice, -1)

        X[i%2, i//2,] = X_slice
        y[i%2, i//2,] = y_slice

    for i in range(len(X)):

        X_heart_locator, _, xc_heart_locator, yc_heart_locator = crop_images(X[i], pred_seg[i])

        stacks_count = math.ceil(X_heart_locator.shape[0] / SLICES)

        pred_mi_heart_locator = np.zeros((X_heart_locator.shape[0], HEIGHT, WIDTH, N_CLASSES))

        for k in range(stacks_count):
            if k == 0 and X_heart_locator.shape[0] < SLICES:
                X_stack = np.zeros((SLICES, ) + X_heart_locator.shape[1:])
                X_stack[(SLICES - X_heart_locator.shape[0]) // 2:X_heart_locator.shape[0] + (SLICES - X_heart_locator.shape[0]) // 2] = X_heart_locator
            elif k < stacks_count - 1:
                X_stack = X_heart_locator[k*SLICES:(k+1)*SLICES]
            else:
                X_stack = X_heart_locator[-SLICES:]

            Xh_heart_locator = np.flip(X_stack, 2)
            Xv_heart_locator = np.flip(X_stack, 1)
            Xhv_heart_locator = np.flip(Xh_heart_locator, 1)

            y_heart_locator_predicted = model.predict(np.expand_dims(X_stack, 0))[0]
            yh_heart_locator_predicted = model.predict(np.expand_dims(Xh_heart_locator, 0))[0]
            yv_heart_locator_predicted = model.predict(np.expand_dims(Xv_heart_locator, 0))[0]
            yhv_heart_locator_predicted = model.predict(np.expand_dims(Xhv_heart_locator, 0))[0]

            yh_heart_locator_predicted = np.flip(yh_heart_locator_predicted, 2)
            yv_heart_locator_predicted = np.flip(yv_heart_locator_predicted, 1)
            yhv_heart_locator_predicted = np.flip(np.flip(yhv_heart_locator_predicted, 2), 1)

            if k == 0 and X_heart_locator.shape[0] < SLICES:
                pred_mi_heart_locator[:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] += ((y_heart_locator_predicted + yh_heart_locator_predicted + yv_heart_locator_predicted + yhv_heart_locator_predicted) / 4)[(SLICES - X_heart_locator.shape[0]) // 2:X_heart_locator.shape[0] + (SLICES - X_heart_locator.shape[0]) // 2, :, :, :]
            elif k < stacks_count - 1:
                pred_mi_heart_locator[k*SLICES:(k+1)*SLICES, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] += (y_heart_locator_predicted + yh_heart_locator_predicted + yv_heart_locator_predicted + yhv_heart_locator_predicted) / 4
            else:
                pred_mi_heart_locator[-SLICES:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] += (y_heart_locator_predicted + yh_heart_locator_predicted + yv_heart_locator_predicted + yhv_heart_locator_predicted) / 4

        if pred_multi_input_heart_locator is None:
            pred_multi_input_heart_locator = pred_mi_heart_locator
        else:
            pred_multi_input_heart_locator = np.concatenate([pred_multi_input_heart_locator, pred_mi_heart_locator], axis=0)

    np.save('./pred-3dn/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', pred_multi_input_heart_locator)
