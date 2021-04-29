import json
import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


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


def mask_image_to_class(mask):
    classes = np.zeros((mask.shape[0], mask.shape[1]))
    idx = np.sum(np.round(mask), axis=-1) > 0
    classes[idx] = (np.argmax(mask, axis=-1) + 1)[idx]
    return classes


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


# Data Generator class
class DataGenerator(Sequence):

    def __init__(self, data_dir='./output/', data_type='train', to_fit=True, batch_size=16,
                 original_height=352, original_width=352, height=320, width=320, n_channels=1, n_classes=4,
                 shuffle=True, custom_keys=[]):
        self.data_dir = data_dir
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (height, width)
        self.height = height
        self.width = width
        self.original_height = original_height
        self.original_width = original_width
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.custom_keys = custom_keys
        self.dataset = self._get_data(data_type)
        self.epoch_count = 0
        self.on_epoch_end()

    # Returns the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))

    # Generates one batch of data
    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self._get_batch_indexes(index)

        # Create temp data list
        list_temp = [self.dataset[k] for k in indexes]
        x_offset = np.random.randint(-10, 10, size=len(list_temp))
        y_offset = np.random.randint(-10, 10, size=len(list_temp))

        # Generate X data
        X = self._generate_X(list_temp, x_offset, y_offset)

        # If it is training, also generates y data
        if self.to_fit:
            y = self._generate_y(list_temp, x_offset, y_offset)
            return X, y
        else:
            return X

    # Returns one batch of indexes
    def _get_batch_indexes(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return indexes

    # Update and shuffle indexes
    def on_epoch_end(self):
        self.epoch_count += 1
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates a batch of X data
    def _generate_X(self, list_temp, x_offset, y_offset):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, sample in enumerate(list_temp):
            # Store sample
            X[i,] = self._crop_image((np.array(Image.open(sample['data'], 'r')) / 127.5 - 1)[:, :, :1], sample['xc'] + x_offset[i], sample['yc'] + y_offset[i])
        return X

    def _generate_y(self, list_temp, x_offset, y_offset):
        y = np.empty((self.batch_size, *self.dim, self.n_classes))
        # Generate data
        for i, sample in enumerate(list_temp):
            # Store mask
            y[i,] = self._crop_image(self._convert_seg_image_to_one_hot_encoding(mask_image_to_class(np.array(Image.open(sample['mask'], 'r')) / 255.)), sample['xc'] + x_offset[i], sample['yc'] + y_offset[i])
        return y

    def _convert_seg_image_to_one_hot_encoding(self, image):
        '''
        image must be either (x, y, z) or (x, y)
        Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
        Example (3D): if input is of shape (x, y, z), the output will ne of shape (x, y, z, n_classes)
        '''
        classes = np.arange(self.n_classes)
        out_image = np.zeros(list(image.shape) + [len(classes)], dtype=image.dtype)
        for i, c in enumerate(classes):
            x = np.zeros((len(classes)))
            x[i] = 1
            out_image[image == c] = x
        return out_image

    def _get_center_point(self, mask):
        x = list()
        y = list()

        for m in mask:
            xc, yc, no_mask = get_center_point(m)
            if not no_mask:
                x.append(xc)
                y.append(yc)

        if len(x) == 0:
            x.append(self.original_width // 2)
            y.append(self.original_height // 2)

        xc = np.round(np.mean(x)).astype(np.int)
        yc = np.round(np.mean(y)).astype(np.int)

        return xc, yc

    def _crop_images(self, data, mask, xc, yc):
        return data[:, yc - self.height // 2:yc + self.height // 2, xc - self.width // 2:xc + self.width // 2, :], mask[:, yc - self.height // 2:yc + self.height // 2, xc - self.width // 2:xc + self.width // 2]

    def _crop_image(self, data, xc, yc):
        return data[yc - self.height // 2:yc + self.height // 2, xc - self.width // 2:xc + self.width // 2, :]


    def _get_data(self, type):
        dataset = []
        with open(self.data_dir + 'ds.json') as json_file:
            ds = json.load(json_file)
        if type == 'custom':
            keys = self.custom_keys
        else:
            keys = ds.keys()

        for key in keys:
            for inst in ds[str(key)]:
                mask = list()
                for i in range(len(inst['ed'])):
                    mask.append(mask_image_to_class(np.array(Image.open(self.data_dir + inst['ed'][i]['gt'], 'r')) / 255.))

                xc, yc = self._get_center_point(mask)
                for i in range(len(inst['ed'])):
                    dataset.append({
                        'data': self.data_dir + inst['ed'][i]['data'],
                        'mask': self.data_dir + inst['ed'][i]['gt'],
                        'xc': xc,
                        'yc': yc
                    })
                # data = list()
                mask = list()
                for i in range(len(inst['es'])):
                    mask.append(mask_image_to_class(np.array(Image.open(self.data_dir + inst['es'][i]['gt'], 'r')) / 255.))

                xc, yc = self._get_center_point(mask)
                for i in range(len(inst['es'])):
                    dataset.append({
                        'data': self.data_dir + inst['es'][i]['data'],
                        'mask': self.data_dir + inst['es'][i]['gt'],
                        'xc': xc,
                        'yc': yc
                    })

        if type == 'val':
            return dataset[int(len(dataset) * 0.8):]
        elif type == 'train':
            return dataset[:int(len(dataset) * 0.8)]
        else:
            return dataset
