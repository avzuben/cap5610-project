import os

import numpy as np
from PIL import Image

from pathlib import Path

from model import get_unet_hl
from data_hl import DataGenerator

if 'VAL_SET' in os.environ:
    VAL_SET = os.environ['VAL_SET']
else:
    VAL_SET = '1'

DATA_DIR = './ds/'
BATCH_SIZE = 10
HEIGHT = 352
WIDTH = 352
N_CHANNELS = 1
N_CLASSES = 1
EPOCHS = 300

val_patients = np.arange(101, 151, 1)

# Create model
model = get_unet_hl(
    input_shape=(HEIGHT, WIDTH, N_CHANNELS),
    n_class=N_CLASSES
)
model.summary()


weight_path = './output-hl/' + VAL_SET + '/best-weights'

model.load_weights(weight_path)


def preprocess_input(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.concatenate((img, img, img), -1)
    return img


def mask_to_image(mask):
    mask = mask.reshape(mask.shape[:-1])
    img = np.zeros(mask.shape + (3,))
    # First class
    idx = np.round(mask) == 1
    img[idx] = 1
    return img


def save_image(img, pat, name):
    img *= 255
    im = Image.fromarray(img.astype(np.uint8))
    im.convert('RGB').save('./output-hl/' + VAL_SET + '/' + pat + '/' + name + '.jpg', 'JPEG')


for pat in val_patients:
    Path('./output-hl/' + VAL_SET + '/' + str(pat).zfill(3)).mkdir(parents=True, exist_ok=True)
    generator = DataGenerator(data_type='test', data_dir=DATA_DIR, batch_size=1,
                              height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                              custom_keys=[pat], shuffle=False, to_fit=False)

    pred = np.zeros((generator.__len__(), HEIGHT, WIDTH, N_CLASSES))
    for i in range(generator.__len__()):
        X = generator.__getitem__(i)
        y_predicted = model.predict(X)

        pred[i, ] = y_predicted[0]
        save_image(preprocess_input(X[0]), str(pat).zfill(3), str(i).zfill(3) + '-input')
        save_image(mask_to_image(y_predicted[0]), str(pat).zfill(3), str(i).zfill(3) + '-pred')

    np.save('./output-hl/' + VAL_SET + '/' + str(pat).zfill(3) + '.npy', pred)
