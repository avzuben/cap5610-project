import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.keras.optimizers import Adam

from network_2d import build_unet
from network_3d import build_unet_3d
from network_2dpp import build_unet_pp


def dice_coeff(y_true, y_pred):
    smooth = 1
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def cce_dice_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def soft_dice(y_true, y_pred):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersect = tf.reduce_sum(y_true * y_pred, axis=0)
    denominator = tf.reduce_sum(y_pred, axis=0) + tf.reduce_sum(y_true, axis=0)
    dice_scores = (2. * intersect) / (denominator + 1e-6)
    return dice_scores


def soft_dice_coeff(y_true, y_pred):
    return tf.reduce_mean(soft_dice(y_true, y_pred))


def hard_dice(y_true, y_pred, class_value):
    # y_true must be label map, not one hot encoding
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    y_true_val = tf.cast(tf.math.equal(y_true, class_value), tf.float32)
    y_pred_val = tf.cast(tf.math.equal(y_pred, class_value), tf.float32)

    return (2. * tf.reduce_sum(y_true_val * y_pred_val) + 1e-7) / (tf.reduce_sum(y_true_val) + tf.reduce_sum(y_pred_val) + 1e-7)


def hard_dice_0(y_true, y_pred):
    return hard_dice(y_true, y_pred, 0)


def hard_dice_1(y_true, y_pred):
    return hard_dice(y_true, y_pred, 1)


def hard_dice_2(y_true, y_pred):
    return hard_dice(y_true, y_pred, 2)


def hard_dice_3(y_true, y_pred):
    return hard_dice(y_true, y_pred, 3)


def accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    return tf.reduce_mean(tf.cast(tf.math.equal(y_true, y_pred), tf.float32))


def get_unet(
        input_shape=(352, 352, 1),
        num_filters=48,
        loss=cce_dice_loss,
        n_class=4,
        dropout=0.3
):

    net, out = build_unet(n_input_channels=input_shape[2], n_output_classes=n_class, input_dim=input_shape[0:2], base_n_filters=num_filters, dropout=dropout)

    model = models.Model(inputs=[net['input']], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss, metrics=[dice_coeff, soft_dice_coeff, hard_dice_0, hard_dice_1, hard_dice_2, hard_dice_3])

    return model


def get_unet_pp(
        input_shape=(352, 352, 1),
        num_filters=48,
        loss=cce_dice_loss,
        n_class=4,
        dropout=0.3
):

    net, out = build_unet_pp(n_input_channels=input_shape[2], n_output_classes=n_class, input_dim=input_shape[0:2], base_n_filters=num_filters, dropout=dropout)

    model = models.Model(inputs=[net['input']], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss, metrics=[dice_coeff, soft_dice_coeff, hard_dice_0, hard_dice_1, hard_dice_2, hard_dice_3])

    return model


def get_unet_hl(
        input_shape=(352, 352, 1),
        num_filters=48,
        loss=bce_dice_loss,
        n_class=1,
        dropout=0.3
):

    net, out = build_unet(n_input_channels=input_shape[2], n_output_classes=n_class, input_dim=input_shape[0:2], base_n_filters=num_filters, dropout=dropout, seg=True)

    model = models.Model(inputs=[net['input']], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss, metrics=[dice_coeff, soft_dice_coeff])

    return model


def get_unet_3d(
        input_shape=(10, 224, 224, 1),
        num_filters=26,
        loss=cce_dice_loss,
        n_class=4,
        dropout=0.3
):

    net, out = build_unet_3d(n_input_channels=input_shape[3], n_output_classes=n_class, input_dim=input_shape[0:3], base_n_filters=num_filters, dropout=dropout)

    model = models.Model(inputs=[net['input']], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss, metrics=[dice_coeff, soft_dice_coeff, hard_dice_0, hard_dice_1, hard_dice_2, hard_dice_3])

    return model


if __name__ == "__main__":

    SLICES = 10
    HEIGHT = 144
    WIDTH = 144
    N_CHANNELS = 1
    N_CLASSES = 4

    model = get_unet_3d(
        input_shape=(SLICES, HEIGHT, WIDTH, N_CHANNELS),
        n_class=N_CLASSES
    )
    model.summary()


