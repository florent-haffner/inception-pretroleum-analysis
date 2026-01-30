import tensorflow as tf
from tensorflow.keras import Model, initializers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, LeakyReLU, Input


class BasicConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(BasicConv1D, self).__init__()
        self.conv = Conv1D(filters, kernel_size=kernel_size, strides=strides, **kwargs)
        self.activation = LeakyReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Module_35x35(tf.keras.layers.Layer):
    def __init__(self, in_channels: int, regularization_factor: float, seed_value: int):
        super(Module_35x35, self).__init__()
        self.branch1 = tf.keras.Sequential([
            MaxPooling1D(pool_size=2),
            BasicConv1D(filters=in_channels * 2,
                        kernel_size=1,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value))
        ])
        self.branch2 = tf.keras.Sequential([
            BasicConv1D(filters=in_channels,
                        kernel_size=1,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
            BasicConv1D(filters=in_channels * 2,
                        kernel_size=3,
                        strides=1,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
        ])
        self.branch3 = tf.keras.Sequential([
            BasicConv1D(filters=in_channels,
                        kernel_size=1,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
            BasicConv1D(filters=in_channels * 2,
                        kernel_size=3,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
            BasicConv1D(filters=in_channels * 2,
                        kernel_size=3,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
        ])
        self.branch4 = tf.keras.Sequential([
            BasicConv1D(filters=in_channels * 2,
                        kernel_size=1,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
        ])

    def call(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = tf.concat([branch1, branch2, branch3, branch4], axis=1)
        return out


class IPA(tf.keras.Model):
    def __init__(self,
                 seed_value,
                 regularization_factor,
                 dropout_rate=0.2):
        super(IPA, self).__init__()
        self.stem = tf.keras.Sequential([
            BasicConv1D(filters=16,
                        kernel_size=3,
                        strides=2,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
            BasicConv1D(filters=16,
                        kernel_size=3,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
            BasicConv1D(filters=16,
                        kernel_size=3,
                        kernel_regularizer=L2(regularization_factor),
                        kernel_initializer=initializers.HeNormal(seed_value)),
        ])
        self.module_35x35 = Module_35x35(in_channels=32,
                                         regularization_factor=regularization_factor,
                                         seed_value=seed_value)
        self.flatten = Flatten()
        self.dropout = Dropout(rate=dropout_rate)
        self.regressor = Dense(1)

    def call(self, x):
        out = self.stem(x)
        out = self.module_35x35(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.regressor(out)
        return out


if __name__ == '__main__':
    # Create an instance of the model
    ipa_model = IPA(seed_value=1, regularization_factor=.0095)

    # Display the model summary
    ipa_model.build((None, 3890, 1))  # Input shape based on your data
    ipa_model.summary()
