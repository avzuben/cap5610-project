import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from PIL import Image

from pathlib import Path

from model import get_unet_hl
from data_hl import DataGenerator

if 'VAL_SET' in os.environ:
    VAL_SET = os.environ['VAL_SET']
else:
    VAL_SET = '5'

DATA_DIR = './ds/'
BATCH_SIZE = 10
HEIGHT = 352
WIDTH = 352
N_CHANNELS = 1
N_CLASSES = 1
EPOCHS = 300

val_patients = np.arange(int(VAL_SET), 101, 5)
train_patients = np.arange(1, 101)
train_patients = np.delete(train_patients, val_patients - 1)

# Create model
model = get_unet_hl(
    input_shape=(HEIGHT, WIDTH, N_CHANNELS),
    n_class=N_CLASSES
)
model.summary()

# Create train and validation generators
train_generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                                custom_keys=train_patients, apply_augmentation=True)
val_generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                              height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                              custom_keys=val_patients)


# Model callbacks
def scheduler(epoch):
    return 0.0005 * (0.985 ** epoch)


weight_path = './output-hl/' + VAL_SET + '/best-weights'

callbacks = [LearningRateScheduler(scheduler, verbose=1),
             ModelCheckpoint(monitor='val_loss',
                             filepath=weight_path,
                             save_weights_only=True,
                             save_best_only=True,
                             mode='min', verbose=1)]

# Model training
history = model.fit(x=train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=callbacks)

# Save weights and history
np.save('./output-hl/' + VAL_SET + '/last-weights', model.get_weights())
np.save('./output-hl/' + VAL_SET + '/history', history.history)

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
    generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=1,
                              height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                              custom_keys=[pat], shuffle=False)

    pred = np.zeros((generator.__len__(), HEIGHT, WIDTH, N_CLASSES))
    for i in range(generator.__len__()):
        X, y = generator.__getitem__(i)
        y_predicted = model.predict(X)

        pred[i, ] = y_predicted[0]
        save_image(preprocess_input(X[0]), str(pat).zfill(3), str(i).zfill(3) + '-input')
        save_image(mask_to_image(y[0]), str(pat).zfill(3), str(i).zfill(3) + '-mask')
        save_image(mask_to_image(y_predicted[0]), str(pat).zfill(3), str(i).zfill(3) + '-pred')

    np.save('./output-hl/' + VAL_SET + '/' + str(pat).zfill(3) + '.npy', pred)
