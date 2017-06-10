import tensorflow as tf

node1 = tf.constant(3, tf.float32)
node2 = tf.constant(3.0)  # default in float32
node3 = tf.constant(3.0)

print(node1)
print(node2)
print(node3)

sess = tf.Session()
print(sess.run([node1, node2, node3]))  #

node4 = tf.multiply(node1, node2)
print(sess.run(node4))
print(node4)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(adder_node)


print(sess.run(adder_node, {a: 3.0, b: 5.0}))
print(sess.run(adder_node, {a: [1.0, 3.4], b: [2.0, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: [1, 2], b: [1, 2]}))
