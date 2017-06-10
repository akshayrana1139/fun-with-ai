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
# Printing Loss compared to the values of "linear_model" and "y" : Value = 0
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# To reduce the loss from (23 -> 0) we need to adjust values of "W" and "b"

# 1. DOING IT MANUALLY

# Fixing the right value of W and b to get 0 loss
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
# Printing loss now, and it seems to be 0 after fixing the "W" and "b"
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# We guessed the "perfect" values of W and b
# but the whole point of machine learning is to find the correct model parameters automatically

# 2. DOING IT AUTOMATICALLY

# Training the model to identify the correct W and b
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# reset to incorrect values
sess.run(init)
for i in range(1000):  # Running 1000 steps
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([W, b]))
# Values of W = -0.99 and that of b = +0.99 which is very close to our correct values of -1 and +1