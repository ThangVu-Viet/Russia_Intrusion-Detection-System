from keras import Input
from keras.layers import Conv2D, Dropout
from keras.layers import Dense, BatchNormalization, MaxPooling2D, concatenate, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import ReLU, ELU, LeakyReLU
from keras import backend as K
import tensorflow as tf
from keras.utils import plot_model


def seg_relu(x):
    return K.switch(x > 0, x, (x/(tf.math.abs(x) + 1)))


dict_activation = {
    'relu': ReLU(),
    'ELU': ELU(),
    'LeakyReLU': LeakyReLU(),
    'seg_relu': seg_relu
}


def block_stem(x, filter_cnv, pool_size=(2, 2), kernel_size=5, strides=1,
               activation='relu', padding='same', name="Block_Stem"):
    conv_stem = Conv2D(filter_cnv, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)(x)
    conv_stem = MaxPooling2D(pool_size=pool_size)(conv_stem)
    conv_stem = Conv2D(filter_cnv * 2, kernel_size=kernel_size//2, strides=strides, activation=activation, padding=padding)(conv_stem)
    conv_stem = MaxPooling2D(pool_size=pool_size, name=name)(conv_stem)
    return conv_stem


def block_identity(x, filter_block, kernel_size_a=5, kernel_size_b=3, kernel_size_c=1,
                   stride=1, activation='relu', padding='same', name="block_identity"):
    conv_a = Conv2D(filter_block, kernel_size=kernel_size_a, strides=stride, activation=activation, padding=padding)(x)
    conv_a = BatchNormalization()(conv_a)

    conv_b = Conv2D(filter_block, kernel_size=kernel_size_b, strides=stride, activation=activation, padding=padding)(x)
    conv_b = BatchNormalization()(conv_b)

    conv_c = Conv2D(filter_block, kernel_size=kernel_size_c, strides=stride, activation=activation, padding=padding)(x)
    conv_c = BatchNormalization()(conv_c)

    conv_concat = concatenate([conv_a, conv_b, conv_c, x])
    output_block = Activation(activation, name=name)(conv_concat)
    return output_block


def deep_learning_model(input_shape, number_class=2, activation_dense='softmax', activation_block='relu'):
    input_layer = Input(shape=input_shape)
    x_ida = block_identity(input_layer, 16, kernel_size_a=5, kernel_size_b=3, kernel_size_c=1,
                           activation=dict_activation[activation_block], name="block_identity_a")
    x_idb = block_identity(x_ida, 32, kernel_size_a=5, kernel_size_b=3, kernel_size_c=1,
                           activation=dict_activation[activation_block], name="block_identity_b")
    x_idc = block_identity(x_idb, 64, kernel_size_a=5, kernel_size_b=3, kernel_size_c=1,
                           activation=dict_activation[activation_block], name="block_identity_c")
    x_concat = concatenate([x_ida, x_idb, x_idc], name="block_concat")
    x_concat = Activation(dict_activation[activation_block], name="block_activation")(x_concat)
    x = GlobalAveragePooling2D()(x_concat)
    x = Dropout(0.5)(x)
    x = Dense(number_class, activation=activation_dense)(x)
    return Model(inputs=input_layer, outputs=x)


# model = deep_learning_model(input_shape=(13, 1, 1), number_class=4)
# model.summary(expand_nested=True)
# plot_model(model, to_file='../docs/model-deep.png', show_shapes=False)
