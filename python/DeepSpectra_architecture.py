import tensorflow as tf
from tensorflow.keras import Model, initializers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, LeakyReLU, Input

KERNEL_SIZE_1 = 9
KERNEL_SIZE_3 = 7
STRIDE_1 = 5
STRIDE_2 = 3


class BasicConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, activation_slope=0.3, **kwargs):
        super(BasicConv1D, self).__init__()
        self.conv = Conv1D(filters, kernel_size=kernel_size, strides=strides, **kwargs)
        self.activation = LeakyReLU(alpha=activation_slope)

    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Module_35x35(tf.keras.layers.Layer):
    def __init__(self, regularization_factor: float, seed_value: int):
        super(Module_35x35, self).__init__()
        self.branch1 = tf.keras.Sequential([
            MaxPooling1D(pool_size=2),
            BasicConv1D(filters=4,
                        kernel_size=1,
                        strides=STRIDE_2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.VarianceScaling(seed_value))
        ])
        self.branch2 = tf.keras.Sequential([
            BasicConv1D(filters=4,
                        kernel_size=1,
                        strides=STRIDE_2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.VarianceScaling(seed_value)),
            BasicConv1D(filters=4,
                        kernel_size=KERNEL_SIZE_3,
                        strides=STRIDE_2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.VarianceScaling(seed_value)),
        ])
        self.branch3 = tf.keras.Sequential([
            BasicConv1D(filters=4,
                        kernel_size=1,
                        strides=STRIDE_2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.VarianceScaling(seed_value)),
            BasicConv1D(filters=4,
                        kernel_size=KERNEL_SIZE_3,
                        strides=STRIDE_2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.VarianceScaling(seed_value)),
        ])
        self.branch4 = tf.keras.Sequential([
            BasicConv1D(filters=4,
                        kernel_size=1,
                        strides=STRIDE_2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.VarianceScaling(seed_value)),
        ])

    def call(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = tf.concat([branch1, branch2, branch3, branch4], axis=1)
        return out


class DeepSpectra(tf.keras.Model):
    def __init__(self,
                 seed_value,
                 # model_name,
                 regularization_factor,
                 dropout_rate=0.2):
        super(DeepSpectra, self).__init__()
        self.hello = BasicConv1D(filters=8,
                                 kernel_size=KERNEL_SIZE_1,
                                 strides=STRIDE_1,
                                 kernel_regularizer=L2(regularization_factor),
                                 kernel_initializer=initializers.VarianceScaling(seed_value))
        self.module_35x35 = Module_35x35(regularization_factor=regularization_factor,
                                         seed_value=seed_value)
        self.flatten = Flatten()
        self.dropout = Dropout(rate=dropout_rate)
        self.regressor_1 = Dense(16)
        self.regressor_2 = Dense(1)

    def call(self, x):
        out = self.hello(x)
        out = self.module_35x35(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.regressor_1(out)
        out = self.regressor_2(out)
        return out
