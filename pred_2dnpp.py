import os

import numpy as np
from PIL import Image
from skimage.morphology import label

from pathlib import Path

from model import get_unet_pp
from data import DataGenerator

DATA_DIR = './ds/'
BATCH_SIZE = 10
HEIGHT = 352
WIDTH = 352
HEIGHT_UNET = 144
WIDTH_UNET = 144
N_CHANNELS = 1
N_CLASSES = 4
EPOCHS = 300

DIR_SEG = './output-hl/'

# Create model
model = get_unet_pp(
    input_shape=(HEIGHT_UNET, WIDTH_UNET, N_CHANNELS),
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

    weight_path = './output-2dnpp/' + VAL_SET + '/best-weights'

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
        im.convert('RGB').save('./pred-2dnpp/' + VAL_SET + '/' + pat + '/' + name + '.jpg', 'JPEG')


    Path('./pred-2dnpp/' + VAL_SET + '/' + str(pat).zfill(3)).mkdir(parents=True, exist_ok=True)
    generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=1,
                              height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                              custom_keys=[pat], shuffle=False, to_fit=True)

    pred_multi_input_mask = None
    pred_multi_input_heart_locator = None
    pred_multi_input_original = None
    pred_multi_input_horizontal = None
    pred_multi_input_vertical = None
    pred_multi_input_horizontal_vertical = None

    print('Patient ' + str(pat))

    X = np.zeros((2, generator.__len__() // 2, HEIGHT, WIDTH, N_CHANNELS))
    y = np.zeros((2, generator.__len__() // 2, HEIGHT, WIDTH))

    for i in range(generator.__len__()):
        X_slice, y_slice = generator.__getitem__(i)
        y_slice = np.argmax(y_slice, -1)

        X[i%2, i//2,] = X_slice
        y[i%2, i//2,] = y_slice

    for i in range(len(X)):

        X_mask, _, xc_mask, yc_mask = crop_images(X[i], y[i])
        X_heart_locator, _, xc_heart_locator, yc_heart_locator = crop_images(X[i], pred_seg[i])

        Xh_mask = np.flip(X_mask, 2)
        Xv_mask = np.flip(X_mask, 1)
        Xhv_mask = np.flip(Xh_mask, 1)

        y_mask_predicted = model.predict(X_mask)
        yh_mask_predicted = model.predict(Xh_mask)
        yv_mask_predicted = model.predict(Xv_mask)
        yhv_mask_predicted = model.predict(Xhv_mask)

        yh_mask_predicted = np.flip(yh_mask_predicted, 2)
        yv_mask_predicted = np.flip(yv_mask_predicted, 1)
        yhv_mask_predicted = np.flip(np.flip(yhv_mask_predicted, 2), 1)

        pred_mi_mask = np.zeros((y_mask_predicted.shape[0], HEIGHT, WIDTH, y_mask_predicted.shape[3]))
        pred_mi_mask[:, yc_mask - HEIGHT_UNET // 2:yc_mask + HEIGHT_UNET // 2, xc_mask - WIDTH_UNET // 2:xc_mask + WIDTH_UNET // 2, :] = (y_mask_predicted + yh_mask_predicted + yv_mask_predicted + yhv_mask_predicted) / 4

        Xh_heart_locator = np.flip(X_heart_locator, 2)
        Xv_heart_locator = np.flip(X_heart_locator, 1)
        Xhv_heart_locator = np.flip(Xh_heart_locator, 1)

        y_heart_locator_predicted = model.predict(X_heart_locator)
        yh_heart_locator_predicted = model.predict(Xh_heart_locator)
        yv_heart_locator_predicted = model.predict(Xv_heart_locator)
        yhv_heart_locator_predicted = model.predict(Xhv_heart_locator)

        yh_heart_locator_predicted = np.flip(yh_heart_locator_predicted, 2)
        yv_heart_locator_predicted = np.flip(yv_heart_locator_predicted, 1)
        yhv_heart_locator_predicted = np.flip(np.flip(yhv_heart_locator_predicted, 2), 1)

        pred_mi_original = np.zeros((y_heart_locator_predicted.shape[0], HEIGHT, WIDTH, y_heart_locator_predicted.shape[3]))
        pred_mi_original[:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] = y_heart_locator_predicted

        pred_mi_horizontal = np.zeros((y_heart_locator_predicted.shape[0], HEIGHT, WIDTH, y_heart_locator_predicted.shape[3]))
        pred_mi_horizontal[:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] = yh_heart_locator_predicted

        pred_mi_vertical = np.zeros((y_heart_locator_predicted.shape[0], HEIGHT, WIDTH, y_heart_locator_predicted.shape[3]))
        pred_mi_vertical[:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] = yv_heart_locator_predicted

        pred_mi_horizontal_vertical = np.zeros((y_heart_locator_predicted.shape[0], HEIGHT, WIDTH, y_heart_locator_predicted.shape[3]))
        pred_mi_horizontal_vertical[:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] = yhv_heart_locator_predicted

        pred_mi_heart_locator = np.zeros((y_heart_locator_predicted.shape[0], HEIGHT, WIDTH, y_heart_locator_predicted.shape[3]))
        pred_mi_heart_locator[:, yc_heart_locator - HEIGHT_UNET // 2:yc_heart_locator + HEIGHT_UNET // 2, xc_heart_locator - WIDTH_UNET // 2:xc_heart_locator + WIDTH_UNET // 2, :] = (y_heart_locator_predicted + yh_heart_locator_predicted + yv_heart_locator_predicted + yhv_heart_locator_predicted) / 4

        if pred_multi_input_mask is None:
            pred_multi_input_mask = pred_mi_mask
            pred_multi_input_heart_locator = pred_mi_heart_locator
            pred_multi_input_original = pred_mi_original
            pred_multi_input_horizontal = pred_mi_horizontal
            pred_multi_input_vertical = pred_mi_vertical
            pred_multi_input_horizontal_vertical = pred_mi_horizontal_vertical
        else:
            pred_multi_input_mask = np.concatenate([pred_multi_input_mask, pred_mi_mask], axis=0)
            pred_multi_input_heart_locator = np.concatenate([pred_multi_input_heart_locator, pred_mi_heart_locator], axis=0)
            pred_multi_input_original = np.concatenate([pred_multi_input_original, pred_mi_original], axis=0)
            pred_multi_input_horizontal = np.concatenate([pred_multi_input_horizontal, pred_mi_horizontal], axis=0)
            pred_multi_input_vertical = np.concatenate([pred_multi_input_vertical, pred_mi_vertical], axis=0)
            pred_multi_input_horizontal_vertical = np.concatenate([pred_multi_input_horizontal_vertical, pred_mi_horizontal_vertical], axis=0)

    np.save('./flip-tests-t/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-mask.npy', pred_multi_input_mask)
    np.save('./flip-tests-t/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', pred_multi_input_heart_locator)
    np.save('./flip-tests-t/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-original.npy', pred_multi_input_original)
    np.save('./flip-tests-t/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-horizontal.npy', pred_multi_input_horizontal)
    np.save('./flip-tests-t/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-vertical.npy', pred_multi_input_vertical)
    np.save('./flip-tests-t/' + VAL_SET + '/' + str(pat).zfill(3) + '-multi-input-horizontal-vertical.npy', pred_multi_input_horizontal_vertical)
