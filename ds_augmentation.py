import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from PIL import Image

from data_3d import DataGenerator3d


def input_to_image(input_image):
    input_image = (input_image + 1) / 2
    return np.concatenate([input_image, input_image, input_image], -1)


def mask_to_image(mask):
    img = np.zeros(mask.shape[0:2] + (3,))
    # First class
    idx = mask == 1
    img[idx] = [1, 0, 0]
    # Second class
    idx = mask == 2
    img[idx] = [0, 1, 0]
    # Third class
    idx = mask == 3
    img[idx] = [0, 0, 1]
    return img


DATA_DIR = './ds/'
BATCH_SIZE = 1
HEIGHT = 352
WIDTH = 352
N_CHANNELS = 1
N_CLASSES = 4

patients = np.arange(1, 101)

xdim = list()
ydim = list()

for pat in patients:

    generator = DataGenerator3d(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                                custom_keys=[pat], shuffle=False, to_fit=True, apply_augmentation=False,
                                augmentation_probability=1, keep_z=True)

    generator_aug = DataGenerator3d(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                    height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                                    custom_keys=[pat], shuffle=False, to_fit=True, apply_augmentation=True,
                                    augmentation_probability=1, keep_z=True)

    for i in range(generator.__len__()):
        print(str(i) + ' out of ' + str(generator.__len__()))
        X, y = generator.__getitem__(i)
        y = np.argmax(y[0], -1)
        for j in range(len(X[0])):
            input_img = Image.fromarray((input_to_image(X[0][j]) * 255).astype(np.uint8))
            input_img.save('./augmented-ds/' + str(pat).zfill(3) + '_' + str(i).zfill(2) + '_' + str(0).zfill(2) + '_' + str(j).zfill(2) + '.jpg')
            gt_img = Image.fromarray((mask_to_image(y[j]) * 255).astype(np.uint8))
            gt_img.save('./augmented-ds/' + str(pat).zfill(3) + '_' + str(i).zfill(2) + '_' + str(0).zfill(2) + '_' + str(j).zfill(2) + '_gt.jpg')
            mask = y[j]
            if len(mask[mask != 0]):
                yp, xp = np.where(mask != 0)
                xdim.append(np.max(xp) - np.min(xp))
                ydim.append(np.max(yp) - np.min(yp))
            else:
                xdim.append(0)
                ydim.append(0)
        for k in range(10):
            X, y = generator_aug.__getitem__(i)
            y = np.argmax(y[0], -1)
            for j in range(len(X[0])):
                input_img = Image.fromarray((input_to_image(X[0][j]) * 255).astype(np.uint8))
                input_img.save('./augmented-ds/' + str(pat).zfill(3) + '_' + str(i).zfill(2) + '_' + str(k + 1).zfill(2) + '_' + str(j).zfill(2) + '.jpg')
                gt_img = Image.fromarray((mask_to_image(y[j]) * 255).astype(np.uint8))
                gt_img.save('./augmented-ds/' + str(pat).zfill(3) + '_' + str(i).zfill(2) + '_' + str(k + 1).zfill(2) + '_' + str(j).zfill(2) + '_gt.jpg')

                mask = y[j]
                if len(mask[mask != 0]):
                    yp, xp = np.where(mask != 0)
                    xdim.append(np.max(xp) - np.min(xp))
                    ydim.append(np.max(yp) - np.min(yp))
                else:
                    xdim.append(0)
                    ydim.append(0)

ifig = 0
fig = plt.figure(ifig)
plt.title('Width distribution')
plt.hist(xdim, 50, density=True)
sns.kdeplot(xdim)
plt.show()

ifig += 1
fig = plt.figure(ifig)
plt.title('Height distribution')
plt.hist(ydim, 50, density=True)
sns.kdeplot(ydim)
plt.show()

ifig += 1
fig = plt.figure(ifig)
plt.ylim([0, 120])
plt.title('Width box')
plt.boxplot(xdim, showfliers=False)
plt.show()

ifig += 1
fig = plt.figure(ifig)
plt.ylim([0, 120])
plt.title('Height box')
plt.boxplot(ydim, showfliers=False)
plt.show()

obj = {}
patients = np.arange(1, 101)

for pat in patients:
    obj[str(pat)] = []
    for i in range(11):
        patient_instance = {'ed': [], 'es': []}
        ed_files = glob.glob('./augmented-ds/' + str(pat).zfill(3) + '_00_' + str(i).zfill(2) + '_*_gt.jpg')
        for f in ed_files:
            patient_instance['ed'].append({
                'data': f.split('\\')[-1].replace('_gt', ''),
                'gt': f.split('\\')[-1]
            })
        es_files = glob.glob('./augmented-ds/' + str(pat).zfill(3) + '_01_' + str(i).zfill(2) + '_*_gt.jpg')
        for f in es_files:
            patient_instance['es'].append({
                'data': f.split('\\')[-1].replace('_gt', ''),
                'gt': f.split('\\')[-1]
            })
        obj[str(pat)].append(patient_instance)

with open('./augmented-ds/ds.json', 'w') as f:
    json.dump(obj, f)
