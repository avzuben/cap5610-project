from collections import OrderedDict

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Add, Reshape, Softmax, Activation, concatenate
from tensorflow.keras import activations


def block(name, n_filters, kernel_size, pad, dropout=None):
    def layer(input_tensor):
        x = input_tensor
        if dropout is not None:
            x = Dropout(dropout, name=name + '_drop')(x)

        x = Conv2D(n_filters, kernel_size, padding=pad, name="{}_1_conv".format(name))(x)
        x = BatchNormalization(name="{}_1_bat_norm".format(name))(x)

        x = Conv2D(n_filters, kernel_size, padding=pad, name="{}_2_conv".format(name))(x)
        x = BatchNormalization(name="{}_2_bat_norm".format(name))(x)

        return x
    return layer


def build_unet_pp(n_input_channels=1, n_output_classes=4, input_dim=(352, 352), base_n_filters=48, dropout=0.3, pad='same', kernel_size=3, seg=False):
    net = OrderedDict()
    name = 'input'
    net[name] = Input((input_dim[0], input_dim[1], n_input_channels), name=name)

    name = 'block_0_0'
    net[name] = block(name, base_n_filters, kernel_size, pad)(net['input'])

    name = 'block_0_0_pool'
    net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net['block_0_0'])

    name = 'block_1_0'
    net[name] = block(name, base_n_filters * 2, kernel_size, pad)(net['block_0_0_pool'])

    name = 'block_1_0_pool'
    net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net['block_1_0'])

    name = 'block_1_0_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_1_0'])

    name = 'block_2_0'
    net[name] = block(name, base_n_filters * 4, kernel_size, pad, dropout)(net['block_1_0_pool'])

    name = 'block_2_0_pool'
    net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net['block_2_0'])

    name = 'block_2_0_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_2_0'])

    name = 'block_3_0'
    net[name] = block(name, base_n_filters * 8, kernel_size, pad, dropout)(net['block_2_0_pool'])

    name = 'block_3_0_pool'
    net[name] = MaxPooling2D((2, 2), strides=(2, 2), name=name)(net['block_3_0'])

    name = 'block_3_0_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_3_0'])

    name = 'block_4_0'
    net[name] = block(name, base_n_filters * 16, kernel_size, pad, dropout)(net['block_3_0_pool'])

    name = 'block_4_0_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_4_0'])

    name = 'block_3_1'
    concat = concatenate([net['block_4_0_upscale'], net['block_3_0']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters * 8, kernel_size, pad, dropout)(concat)

    name = 'block_3_1_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_3_1'])

    name = 'block_2_1'
    concat = concatenate([net['block_3_0_upscale'], net['block_2_0']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters * 4, kernel_size, pad, dropout)(concat)

    name = 'block_2_1_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_2_1'])

    name = 'block_2_2'
    concat = concatenate([net['block_3_1_upscale'], net['block_2_0'], net['block_2_1']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters * 4, kernel_size, pad, dropout)(concat)

    name = 'block_2_2_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_2_2'])

    name = 'block_1_1'
    concat = concatenate([net['block_2_0_upscale'], net['block_1_0']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters * 2, kernel_size, pad, dropout)(concat)

    name = 'block_1_1_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_1_1'])

    name = 'block_1_2'
    concat = concatenate([net['block_2_1_upscale'], net['block_1_0'], net['block_1_1']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters * 2, kernel_size, pad, dropout)(concat)

    name = 'block_1_2_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_1_2'])

    name = 'block_1_3'
    concat = concatenate([net['block_2_2_upscale'], net['block_1_0'], net['block_1_1'], net['block_1_2']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters * 2, kernel_size, pad, dropout)(concat)

    name = 'block_1_3_upscale'
    net[name] = UpSampling2D(size=(2, 2), name=name)(net['block_1_3'])

    name = 'block_0_1'
    concat = concatenate([net['block_1_0_upscale'], net['block_0_0']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters, kernel_size, pad, dropout)(concat)

    name = 'block_0_2'
    concat = concatenate([net['block_1_1_upscale'], net['block_0_0'], net['block_0_1']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters, kernel_size, pad, dropout)(concat)

    name = 'block_0_3'
    concat = concatenate([net['block_1_2_upscale'], net['block_0_0'], net['block_0_1'], net['block_0_2']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters, kernel_size, pad, dropout)(concat)

    name = 'block_0_4'
    concat = concatenate([net['block_1_3_upscale'], net['block_0_0'], net['block_0_1'], net['block_0_2'], net['block_0_3']], axis=-1, name=name + '_concat')
    net[name] = block(name, base_n_filters, kernel_size, pad, dropout)(concat)

    output_0_4 = Conv2D(n_output_classes, 1, name='output_block_0_4')(net['block_0_4'])
    output_0_3 = Conv2D(n_output_classes, 1, name='output_block_0_3')(net['block_0_3'])
    output_0_2 = Conv2D(n_output_classes, 1, name='output_block_0_2')(net['block_0_2'])

    output_sum = Add(name='output_sum')([output_0_4, output_0_3, output_0_2])

    net['reshape_seg'] = Reshape((input_dim[0], input_dim[1], n_output_classes))(output_sum)

    if not seg:
        net['output'] = Softmax()(net['reshape_seg'])
    else:
        net['output'] = Activation(activations.sigmoid)(net['reshape_seg'])

    return net, net['output']
