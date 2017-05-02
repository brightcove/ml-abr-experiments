import tensorflow as tf

NUM_UNITS = 30

# -- model --
x = tf.placeholder(tf.float32, [None, 3], 'x')
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py#L43
weights = tf.Variable(tf.random_normal([NUM_UNITS, 1]))
biases = tf.Variable(tf.random_normal([1]))
lstm = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
outputs, states = tf.contrib.rnn.static_rnn(lstm, [x], dtype=tf.float32)

# the output of the network is the output of the last LSTM cell
# y = outputs[-1][:, -1] * tf.random_normal([1])
y = tf.matmul(outputs[-1], weights) + biases

# -- training --
y_ = tf.placeholder(tf.float32, [None, 1], 'y_')
error = tf.reduce_mean(tf.squared_difference(y_, y))
optimizer = tf.train.AdamOptimizer().minimize(error)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
    session.run(optimizer, feed_dict={
        x: [[1, 2, 3],
             [4, 5, 6],
             [22, 23, 24]],
        y_: [[4], [7], [25]]
    })

# -- eval --
print(session.run([y, error], feed_dict={
    x: [[1, 2, 3],
        [4, 5, 6],
        [22, 23, 24],
        [78, 79, 80],
        [101, 111, 121],
        [11, 12, 13],
        [19, 20, 21],
        [-1, 0, 1]],
    y_: [[4], [7], [25], [81], [131], [14], [22], [2]]
}))
