import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from dataset_utils import load_dataset, generate_test_patient_info


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def mask_to_image(mask):
    img = np.zeros(mask.shape[0:3] + (3,))
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
    classes = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
    idx = np.sum(np.round(mask), axis=-1) > 0
    classes[idx] = (np.argmax(mask, axis=-1) + 1)[idx]
    return classes


# Data Generator class
class DataGenerator3d(Sequence):

    def __init__(self, data_dir='./output-3d/', data_type='train', to_fit=True, batch_size=16,
                 slices=10, height=320, width=320, n_channels=1, n_classes=4, shuffle=True, custom_keys=[],
                 input_norm_dist_percent=0.98, apply_augmentation=False, augmentation_probability=0.67, keep_z=False):
        self.data_dir = data_dir
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (slices, height, width)
        self.keep_z = keep_z
        self.last_slices = slices
        self.slices = slices
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.custom_keys = custom_keys
        self.input_norm_dist_percent = input_norm_dist_percent
        self.apply_augmentation = apply_augmentation
        self.augmentation_probability = augmentation_probability
        self.augmentation_config = {
            'rotation': (0, 360),
            'scale': (0.7, 1.3),
            'alpha': (0, 350),
            'sigma': (14, 17),
        }
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

        if not self.apply_augmentation:

            # Generate X data
            X = self._generate_X(list_temp)

            # If it is training, also generates y data
            if self.to_fit:
                y = self._generate_y(list_temp)
                return X, y

            else:
                return X

        else:
            return self._generate_Xy(list_temp)

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
    def _generate_X(self, list_temp):
        if self.apply_augmentation:
            n_channels = 3
        else:
            n_channels = self.n_channels
        if self.keep_z:
            shape = (self.batch_size, self.last_slices, self.height, self.width, n_channels)
        else:
            shape = (self.batch_size, *self.dim, n_channels)
        X = np.empty(shape)
        # Generate data
        for i, sample in enumerate(list_temp):
            # Store sample
            X[i,] = sample['data']
        return X

    def _generate_y(self, list_temp):
        if self.apply_augmentation:
            n_classes = 3
        else:
            n_classes = self.n_classes
        if self.keep_z:
            shape = (self.batch_size, self.last_slices, self.height, self.width, n_classes)
        else:
            shape = (self.batch_size, *self.dim, n_classes)
        y = np.empty(shape)
        # Generate data
        for i, sample in enumerate(list_temp):
            # Store mask
            y[i,] = sample['mask']
        return y

    def _generate_Xy(self, list_temp):
        if self.keep_z:
            shape_X = (self.batch_size, self.last_slices, self.height, self.width, self.n_channels)
            shape_y = (self.batch_size, self.last_slices, self.height, self.width, self.n_classes)
        else:
            shape_X = (self.batch_size, *self.dim, self.n_channels)
            shape_y = (self.batch_size, *self.dim, self.n_classes)
        X = np.empty(shape_X)
        y = np.empty(shape_y)
        # Generate data
        for i, sample in enumerate(list_temp):
            X[i,], y[i,] = self._data_augmentation(sample['data'], sample['mask'])
        return X, y

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

    def _normalize_input(self, image):
        image = image - image.mean()
        pixels = image.flatten()
        delta_index = int(round(((1 - self.input_norm_dist_percent) / 2) * len(pixels)))
        pixels = np.sort(pixels)
        min = pixels[delta_index]
        max = pixels[-(delta_index + 1)]
        image = 2 * ((image - min) / (max - min)) - 1
        image[image < -1] = -1
        image[image > 1] = 1
        return image

    def _resize_padding(self, image, n_channels=None, pad_value=0, expand_dim=False):
        if self.keep_z:
            slices = image.shape[0]
        else:
            slices = self.slices
        if n_channels is not None:
            data = np.zeros((slices, self.height, self.width, n_channels))
        else:
            data = np.zeros((slices, self.height, self.width))
        data += pad_value
        s_offest = (slices - image.shape[0]) // 2
        h_offest = (self.height - image.shape[1]) // 2
        w_offest = (self.width - image.shape[2]) // 2

        t_s_s = max(s_offest, 0)
        t_s_e = t_s_s + min(image.shape[0] + s_offest, image.shape[0]) - max(0, -s_offest)
        t_h_s = max(h_offest, 0)
        t_h_e = t_h_s + min(image.shape[1] + h_offest, image.shape[1]) - max(0, -h_offest)
        t_w_s = max(w_offest, 0)
        t_w_e = t_w_s + min(image.shape[2] + w_offest, image.shape[2]) - max(0, -w_offest)

        s_s_s = max(0, -s_offest)
        s_s_e = s_s_s + t_s_e - t_s_s
        s_h_s = max(0, -h_offest)
        s_h_e = s_h_s + t_h_e - t_h_s
        s_w_s = max(0, -w_offest)
        s_w_e = s_w_s + t_w_e - t_w_s

        if expand_dim:
            data[t_s_s:t_s_e, t_h_s:t_h_e, t_w_s:t_w_e] = np.expand_dims(image[s_s_s:s_s_e, s_h_s:s_h_e, s_w_s:s_w_e], axis=-1)
        else:
            data[t_s_s:t_s_e, t_h_s:t_h_e, t_w_s:t_w_e] = image[s_s_s:s_s_e, s_h_s:s_h_e, s_w_s:s_w_e]
        return data

    def _rotate_image(self, input_img, label_img):
        rotation = np.random.randint(self.augmentation_config['rotation'][0], self.augmentation_config['rotation'][1] + 1)
        new_input_img = np.zeros(input_img.shape)
        new_label_img = np.zeros(label_img.shape)
        for i in range(input_img.shape[0]):
            new_input_img[i, ] = np.array(Image.fromarray(input_img[i].astype(np.uint8)).rotate(rotation))
            new_label_img[i, ] = np.array(Image.fromarray(label_img[i].astype(np.uint8)).rotate(rotation))
        return new_input_img, new_label_img

    def _scale_image(self, input_img, label_img):
        scale = np.random.random() * (self.augmentation_config['scale'][1] - self.augmentation_config['scale'][0]) + self.augmentation_config['scale'][0]
        height = int(scale * self.height)
        width = int(scale * self.width)
        new_input_img = np.zeros((input_img.shape[0], height, width, input_img.shape[3]))
        new_label_img = np.zeros((input_img.shape[0], height, width, input_img.shape[3]))
        for i in range(input_img.shape[0]):
            new_input_img[i,] = np.array(Image.fromarray(input_img[i].astype(np.uint8)).resize((width, height), Image.NEAREST))
            new_label_img[i,] = np.array(Image.fromarray(label_img[i].astype(np.uint8)).resize((width, height), Image.NEAREST))
        return input_img, label_img

    def _elastic_transform(self, input_img, label_img):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        random_state = np.random.RandomState(None)
        alpha = np.random.randint(self.augmentation_config['alpha'][0], self.augmentation_config['alpha'][1] + 1)
        sigma = np.random.randint(self.augmentation_config['sigma'][0], self.augmentation_config['sigma'][1] + 1)

        shape = input_img.shape[1:]

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        for i in range(input_img.shape[0]):
            input_img[i] = map_coordinates(input_img[i], indices, order=1, mode='reflect').reshape(shape)
            label_img[i] = map_coordinates(label_img[i], indices, order=1, mode='reflect').reshape(shape)

        return input_img, label_img

    def _data_augmentation(self, X, y):
        X = X * 127.5 + 127.5
        # Horizontal flip
        if np.random.uniform() < 0.5:
            X = np.flip(X, 2)
            y = np.flip(y, 2)
        # Vertical flip
        if np.random.uniform() < 0.5:
            X = np.flip(X, 1)
            y = np.flip(y, 1)
        if np.random.uniform() < self.augmentation_probability:
            X, y = self._rotate_image(X, y)
            X, y = self._scale_image(X, y)
            X, y = self._elastic_transform(X, y)
            # pass
        X_new = self._resize_padding(2 * (np.expand_dims(rgb2gray(X), axis=-1) / 255) - 1, self.n_channels, -1)
        y_new = self._convert_seg_image_to_one_hot_encoding(self._resize_padding(mask_image_to_class(y)))
        return X_new, y_new

    def _get_data(self, type):
        if type == 'test':
            return self._get_test_data()
        else:
            return self._get_train_data(type)

    def _get_train_data(self, type):
        dataset = []
        ds = load_dataset(root_dir=self.data_dir)
        if type == 'custom':
            keys = self.custom_keys
        else:
            keys = ds.keys()

        for key in keys:
            self.last_slices = len(ds[key]['ed_data'])
            if self.apply_augmentation:
                data = np.expand_dims(self._normalize_input(ds[key]['ed_data']), axis=-1)
                dataset.append({
                    'data': np.concatenate([data, data, data], axis=-1),
                    'mask': mask_to_image(ds[key]['ed_gt'])
                })

                data = np.expand_dims(self._normalize_input(ds[key]['es_data']), axis=-1)
                dataset.append({
                    'data': np.concatenate([data, data, data], axis=-1),
                    'mask': mask_to_image(ds[key]['es_gt'])
                })
            else:
                dataset.append({
                    'data': self._resize_padding(self._normalize_input(ds[key]['ed_data']), self.n_channels, -1,
                                                 True),
                    'mask': self._convert_seg_image_to_one_hot_encoding(self._resize_padding(ds[key]['ed_gt']))
                })

                dataset.append({
                    'data': self._resize_padding(self._normalize_input(ds[key]['es_data']), self.n_channels, -1,
                                                 True),
                    'mask': self._convert_seg_image_to_one_hot_encoding(self._resize_padding(ds[key]['es_gt']))
                })

        if type == 'val':
            return dataset[int(len(dataset) * 0.8):]
        elif type == 'train':
            return dataset[:int(len(dataset) * 0.8)]
        else:
            return dataset

    def _get_test_data(self):
        dataset = []
        ds = generate_test_patient_info('./testing')
        if len(self.custom_keys) > 0:
            keys = self.custom_keys
        else:
            keys = ds.keys()

        for key in keys:
            dataset.append({
                'data': self._resize_padding(self._normalize_input(ds[key]['ed_data']), self.n_channels, -1, True)
            })

            dataset.append({
                'data': self._resize_padding(self._normalize_input(ds[key]['es_data']), self.n_channels, -1, True)
            })

        return dataset
