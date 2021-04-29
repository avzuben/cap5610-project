from collections import OrderedDict

from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Dropout, UpSampling3D, Add, Reshape, Softmax, concatenate

def build_unet_3d(n_input_channels=1, n_output_classes=4, input_dim=(10, 352, 352), base_n_filters=48, dropout=0.3, pad='same', kernel_size=3):

    net = OrderedDict()

    name = 'input'
    net[name] = Input((input_dim[0], input_dim[1], input_dim[2], n_input_channels), name=name)

    name = 'contr_1_1'
    net[name] = Conv3D(base_n_filters, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_1_2'
    net[name] = Conv3D(base_n_filters, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool1'
    net[name] = MaxPooling3D((1, 2, 2), name=name)(net[prev])

    prev = name
    name = 'contr_2_1'
    net[name] = Conv3D(base_n_filters * 2, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_2_2'
    net[name] = Conv3D(base_n_filters * 2, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool2'
    x = net[name] = MaxPooling3D((1, 2, 2), name=name)(net[prev])

    if dropout is not None:
        x = Dropout(dropout, name='drop2')(x)

    name = 'contr_3_1'
    net[name] = Conv3D(base_n_filters * 4, kernel_size, padding=pad, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_3_2'
    net[name] = Conv3D(base_n_filters * 4, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool3'
    x = net[name] = MaxPooling3D((1, 2, 2), name=name)(net[prev])

    if dropout is not None:
        x = Dropout(dropout, name='drop3')(x)

    name = 'contr_4_1'
    net[name] = Conv3D(base_n_filters * 8, kernel_size, padding=pad, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'contr_4_2'
    net[name] = Conv3D(base_n_filters * 8, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'pool4'
    x = net[name] = MaxPooling3D((1, 2, 2), name=name)(net[prev])

    if dropout is not None:
        x = Dropout(dropout, name='drop4')(x)

    name = 'encode_1'
    net[name] = Conv3D(base_n_filters * 16, kernel_size, padding=pad, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'encode_2'
    net[name] = Conv3D(base_n_filters*16, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale1'
    net[name] = UpSampling3D(size=(1, 2, 2), name=name)(net[prev])

    name = 'concat1'
    x = net[name] = concatenate([net['upscale1'], net['contr_4_2']], axis=-1, name=name)

    if dropout is not None:
        x = Dropout(dropout, name='drop5')(x)

    name = 'expand_1_1'
    net[name] = Conv3D(base_n_filters * 8, kernel_size, padding=pad, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_1_2'
    net[name] = Conv3D(base_n_filters*8, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale2'
    net[name] = UpSampling3D(size=(1, 2, 2), name=name)(net[prev])

    name = 'concat2'
    x = net[name] = concatenate([net['upscale2'], net['contr_3_2']], axis=-1, name=name)

    if dropout is not None:
        x = Dropout(dropout, name='drop6')(x)

    name = 'expand_2_1'
    net[name] = Conv3D(base_n_filters * 4, kernel_size, padding=pad, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_2_2'
    net[name] = Conv3D(base_n_filters*4, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    ds2 = net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale3'
    net[name] = UpSampling3D(size=(1, 2, 2), name=name)(net[prev])

    name = 'concat3'
    x = net[name] = concatenate([net['upscale3'], net['contr_2_2']], axis=-1, name=name)

    if dropout is not None:
        x = Dropout(dropout, name='drop7')(x)

    name = 'expand_3_1'
    net[name] = Conv3D(base_n_filters * 2, kernel_size, padding=pad, name="{}_conv".format(name))(x)
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_3_2'
    net[name] = Conv3D(base_n_filters*2, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'upscale4'
    net[name] = UpSampling3D(size=(1, 2, 2), name=name)(net[prev])

    name = 'concat4'
    net[name] = concatenate([net['upscale4'], net['contr_1_2']], axis=-1, name=name)

    prev = name
    name = 'expand_4_1'
    net[name] = Conv3D(base_n_filters, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'expand_4_2'
    net[name] = Conv3D(base_n_filters, kernel_size, padding=pad, name="{}_conv".format(name))(net[prev])
    net[name] = BatchNormalization(name="{}_bat_norm".format(name))(net[name])

    prev = name
    name = 'output_segmentation'
    net[name] = Conv3D(n_output_classes, 1, name=name)(net[prev])

    ds2_1x1_conv = Conv3D(n_output_classes, 1, padding='same', name='ds2_1x1_conv')(ds2)

    ds1_ds2_sum_upscale = UpSampling3D(size=(1, 2, 2), name='ds1_ds2_sum_upscale')(ds2_1x1_conv)

    ds3_1x1_conv = Conv3D(n_output_classes, 1, padding='same', name='ds3_1x1_conv')(net['expand_3_2'])

    ds1_ds2_sum_upscale_ds3_sum = Add(name='ds1_ds2_sum_upscale_ds3_sum')([ds1_ds2_sum_upscale, ds3_1x1_conv])
    ds1_ds2_sum_upscale_ds3_sum_upscale = UpSampling3D(size=(1, 2, 2), name='ds1_ds2_sum_upscale_ds3_sum_upscale')(ds1_ds2_sum_upscale_ds3_sum)

    seg_layer = Add(name='seg')([net['output_segmentation'], ds1_ds2_sum_upscale_ds3_sum_upscale])

    net['reshapeSeg'] = Reshape((input_dim[0], input_dim[1], input_dim[2], n_output_classes))(seg_layer)

    net['output'] = Softmax()(net['reshapeSeg'])

    return net, net['output']
