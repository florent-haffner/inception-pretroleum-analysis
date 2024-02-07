#!/usr/bin/env python
# coding: utf-8
import os.path
import time
import warnings
warnings.filterwarnings("ignore")

from scipy.io import loadmat
import numpy as np
from numpy.random import seed
import seaborn as sns
sns.set_theme(style="ticks", palette='muted')

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import optimizers
from tqdm.keras import TqdmCallback

from python.DeepSpectra_architecture import DeepSpectra
from python.plot_utils import plot_history, plot_val, plot_err
from python.regression_utils import cnn_prediction

SEED_VALUE = 1
seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
tf.keras.utils.set_random_seed(SEED_VALUE)

print(f'TF v {tf.__version__} & Keras v {tf.keras.__version__}')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Memory growth activated')

"""
Data loading
---
Change here the path to access the dataset.
Dataset name logic: {raw/processed}-{split_type}-{spectral_range_description}
- "data/raw-kennard-reduced_range.mat" 
- "data/raw-kennard-full_range.mat" 
"""
X = loadmat("data/raw-kennard-reduced_range.mat")
# X = loadmat("data/raw-kennard-full_range.mat")

X_train = X['X_train']
X_test = X['X_test']

y_train_scaled = np.genfromtxt('data/y_train_scaled.csv', delimiter=',')
y_test_scaled = np.genfromtxt('data/y_test_scaled.csv', delimiter=',')
y_train_scaled = y_train_scaled[..., np.newaxis]
y_test_scaled = y_test_scaled[..., np.newaxis]
"""
print(X_train.shape, y_train_scaled.shape, X_test.shape, y_test_scaled.shape)
>> (174, 3890) (174, 1) (75, 3890) (75, 1)
"""

scalerX = StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)
X_train_reshaped = X_train_scaled[..., np.newaxis]
X_test_reshaped = X_test_scaled[..., np.newaxis]
"""
print(X_train_reshaped.shape, y_train_scaled.shape, X_test_reshaped.shape, y_test_scaled.shape)
>> (174, 3890, 1) (174, 1) (75, 3890, 1) (75, 1)
"""


""" MODEL LOGISTICS """
MODEL_NAME = 'DeepSpectra'
path_model = 'model/'

""" BASE HPs """
LEARNING_RATE = .01
BATCH_SIZE = 32
EPOCHS = 500

""" REGULARIZATION HPs """
REGULARIZATION_COEF = .001
DROPOUT_RATE = .2

if not os.path.exists(path_model + MODEL_NAME):
    print('Model does not exists, training in progress...')
    """
    # Model input
    n_wavenumber = X_train_reshaped.shape[1]
    fill_dimension = 1
    input_spectra = Input(shape=(n_wavenumber, fill_dimension))
    >> (3890, 1)
    """
    model = DeepSpectra(seed_value=SEED_VALUE,
                        regularization_factor=REGULARIZATION_COEF,
                        dropout_rate=DROPOUT_RATE)

    # Optimization parameters
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=10000, decay_rate=.001)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

    # Performing model training
    # It take about 35s
    history = model.fit(
        X_train_reshaped, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_reshaped, y_test_scaled),
        callbacks=[stop_early, TqdmCallback(verbose=0)],
        verbose=0,  # Suppress logging.
    )

    # Regression
    RMSEP, R2P, y_preds = cnn_prediction(model, X_train_reshaped, X_test_reshaped,
                                         y_train_scaled, y_test_scaled)
    plot_history(history)
    print('Saving model')
    model.save(path_model + MODEL_NAME, overwrite=True, save_format='tf')

else:
    print('Model does exist, performing regression.')
    model = tf.keras.models.load_model(path_model + MODEL_NAME)

    # Regression
    _, _, y_preds = cnn_prediction(model, X_train_reshaped, X_test_reshaped, y_train_scaled, y_test_scaled)

model.summary()
