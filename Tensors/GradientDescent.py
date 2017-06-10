import tensorflow as tf
import os
# For disabling messages for CPU computations.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
sess = tf.Session()

# Used to initialize tf.Variable ( tf.constants are autoinitialized when called)
init = tf.global_variables_initializer()
# init is a handle to tf sub-graph. It is initialized with the below command.
sess.run(init)
# Evaluating linear model
print(sess.run(linear_model, {x: [1, 2, 3, 4, 5]}))

# Creating a placeholder for desired value.
y = tf.placeholder(tf.float32)
# Creating a mean squared loss function
squared_deltas = tf.square(linear_model - y) 
loss = tf.reduce_sum(squared_deltas)
# Printing Loss compared to the values of "linear_model" and "y"
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))



