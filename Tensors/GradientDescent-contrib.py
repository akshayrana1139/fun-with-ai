import numpy as np
import tensorflow as tf
import os

# For disabling messages for CPU computations.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# We have only one real valued feature`
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation (inference)
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Reading and setting up data with how many batches of data (num_epochs) and how big each batch
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)
# Trained on input_fn
estimator.fit(input_fn=input_fn, steps=1000)

# Testing on input_fn
print(estimator.evaluate(input_fn=input_fn))
                                        