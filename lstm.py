# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

NUM_UNITS = 30
MAX_SAMPLES = 4
SAMPLES_PER_STEP = 1

def dynamic_index(outputs, sequence_length):
    """
    Create a TensorFlow op to collect the output corresponding to
    sequence_length. This is a workaround because TensorFlow does not
    support dynamic indexing. See https://git.io/dynamic-rnn-indexing
    """
    # transform outputs into shape [batch_size, MAX_SAMPLES,
    # NUM_UNITS] and grab the batch_size
    stacked = tf.stack(outputs)
    transposed = tf.transpose(stacked, [1, 0, 2])
    batch_size = tf.shape(transposed)[0]

    # our network produces one output for every bitrate sample in input
    # since the number of samples in each input is variable, we have
    # to calculate the output position of the final prediction
    # dynamically
    index = (tf.range(0, batch_size) * MAX_SAMPLES) + (sequence_length - 1)
    # reshaped has shape [batch_size * MAX_SAMPLES, NUM_UNITS] and we
    # use index to pick out the output corresponding to
    # sequence_length for each input
    reshaped = tf.reshape(transposed, [-1, NUM_UNITS])
    return tf.gather(reshaped, index)

# -- model --
# x => [batch size, max_samples, samples_per_step]
x = tf.placeholder(tf.float32, [None, MAX_SAMPLES, SAMPLES_PER_STEP], 'x')
sequence_length = tf.placeholder(tf.int32, [None])
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py#L43
weights = tf.Variable(tf.random_normal([NUM_UNITS, 1]))
biases = tf.Variable(tf.random_normal([1]))

lstm = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
outputs, states = tf.contrib.rnn.static_rnn(lstm,
                                            tf.unstack(x, MAX_SAMPLES, SAMPLES_PER_STEP),
                                            sequence_length=sequence_length,
                                            dtype=tf.float32)

y = tf.matmul(dynamic_index(outputs, sequence_length), weights) + biases

# -- training --
y_ = tf.placeholder(tf.float32, [None, 1], 'y_')
error = tf.reduce_mean(tf.squared_difference(y_, y))
optimizer = tf.train.AdamOptimizer().minimize(error)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
    session.run(optimizer, feed_dict={
        x: [[[1], [2], [3], [4]],
            [[4], [5], [6], [-1]],
            [[22], [23], [24], [-1]],
            [[7], [-1], [-1], [-1]]],
        sequence_length: [4, 3, 3, 1],
        y_: [[5], [7], [25], [8]]
    })

# -- eval --
prediction, actual, mean_error = session.run([y, y_, error], feed_dict={
    x: [[[1], [2], [3], [-1]],
        [[4], [5], [6], [-1]],
        [[22], [23], [24], [-1]],
        [[78], [79], [80], [-1]],
        [[101], [111], [121], [131]],
        [[11], [12], [-1], [-1]],
        [[19], [20], [21], [-1]],
        [[-1], [0], [-1], [-1]],
        [[10], [-1], [-1], [-1]]],
    sequence_length: [3, 3, 3, 3, 4, 2, 3, 2, 1],
    y_: [[4], [7], [25], [81], [141], [13], [22], [1], [11]]
})
print '--- Results ---'
print 'Overall Error: %f' % mean_error
for i in range(0, len(prediction)):
    print 'predicted: %.1f, actual: %.1f, difference: %.1f' % (prediction[i], actual[i], actual[i] - prediction[i])
