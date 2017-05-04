# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
import numpy as np
import tensorflow as tf
import argparse
import os
import random
import sys
import fileinput

tf.logging.set_verbosity(tf.logging.INFO)

NUM_UNITS = 30
MAX_SAMPLES = 4
SAMPLES_PER_STEP = 1
bps_to_MBps = 1 / (8.0 * 1024 * 1024)

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

def parse_input(file_path):
    """
    Turn an input file of newline-separate bitrate samples into
    input and label arrays. An input file line should look like this:
      4983 1008073 1591538 704983 1008073 1008073 704983
    Adjacent duplicate entries will be removed and lines with less than
    two samples will be filtered out.

    @return a tuple of the x, x sequence length, and y arrays parsed
    from the input file
    """
    bitrate_inputs = []
    inputs_length = []
    bitrate_labels = []
    with open(file_path, 'r') as file:
        for line in file:
            samples = map(lambda x: [float(x) * bps_to_MBps], line.strip().split(' '))[0:MAX_SAMPLES + 1]
            if (len(samples) < 2):
                # skip lines without enough samples
                continue
            bitrate_labels.append(samples.pop())
            inputs_length.append(len(samples))
            samples += [[-1] for i in range(MAX_SAMPLES - len(samples))]
            bitrate_inputs += [samples]
    return bitrate_inputs, inputs_length, bitrate_labels

# process command-line arguments
cli = argparse.ArgumentParser(description='Train and evaluate an RNN.')
cli.add_argument('--train',
                 dest='training_mode',
                 action='store_true',
                 help='Train the RNN against the TRAINING_FILE')
cli.add_argument('--training-data',
                 dest='training_file',
                 default='data/bitrate_samples_0.training',
                 help='The path to training bitrate sample data')
cli.add_argument('--validation-data',
                 dest='validation_file',
                 default='data/bitrate_samples_0.validation',
                 help='The path to validation bitrate sample data')
cli.add_argument('--save',
                 dest='output_file',
                 default='lstm_checkpoint',
                 help='Save the trained RNN to the specified file')
cli.add_argument('--load',
                 dest='checkpoint',
                 default='lstm_checkpoint',
                 help='When not training, the pre-trained RNN to load')

arguments = cli.parse_args()
if (arguments.training_file == None
    or not os.path.lexists(arguments.training_file)):
    print 'Could not open training data at "%s"' & arguments.training_file
    sys.exit(1)
if (arguments.validation_file == None
    or not os.path.lexists(arguments.validation_file)):
    print 'Could not open training data at "%s"' & arguments.validation_file
    sys.exit(1)

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
y_ = tf.placeholder(tf.float32, [None, 1], 'y_')
error = tf.reduce_mean(tf.squared_difference(y_, y))
optimizer = tf.train.AdamOptimizer().minimize(error)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

# -- training --
if (arguments.training_mode):
    # read input data
    print 'Reading training data from "%s"' % arguments.training_file
    try:
        training_x, training_length, training_y_ = parse_input(arguments.training_file)
    except IOError:
        print 'Error reading training file'
        sys.exit(1)

    print 'x: %s\nsequence_length: %s\ny_: %s' % (training_x[0:5], training_length[0:5], training_y_[0:5])
    print 'Training...'
    session.run(optimizer, feed_dict={
        x: training_x,
        sequence_length: training_length,
        y_: training_y_
    })

    # save the trained network to disk
    saver.save(session, arguments.output_file)
    print 'Saved "%s".' % arguments.output_file
else:
    saver.restore(session, arguments.checkpoint)

# -- eval --
# read validation data
print 'Reading validation data from "%s"' % arguments.validation_file
try:
    validation_x, validation_length, validation_y_ = parse_input(arguments.validation_file)
except IOError:
    print 'Error reading training file "%s"' % arguments.training_file
    sys.exit(1)

print 'Validating...'
prediction, actual, mean_error = session.run([y, y_, error], feed_dict={
    # x: [[[1], [2], [3], [-1]],
    #     [[4], [5], [6], [-1]],
    #     [[22], [23], [24], [-1]],
    #     [[78], [79], [80], [-1]],
    #     [[101], [111], [121], [131]],
    #     [[11], [12], [-1], [-1]],
    #     [[19], [20], [21], [-1]],
    #     [[-1], [0], [-1], [-1]],
    #     [[10], [-1], [-1], [-1]]],
    # sequence_length: [3, 3, 3, 3, 4, 2, 3, 2, 1],
    # y_: [[4], [7], [25], [81], [141], [13], [22], [1], [11]]

    # x: validation_x[0:5],
    # sequence_length: validation_length[0:5],
    # y_: validation_y_[0:5]

    x: validation_x,
    sequence_length: validation_length,
    y_: validation_y_
})

print '--- Results ---'
print 'Overall Error: %f' % mean_error
sample_indices = sorted(random.sample(xrange(len(prediction)), 10))
for i in sample_indices:
    print 'predicted: %.1f, actual: %.1f, difference: %.1f' % (prediction[i], actual[i], actual[i] - prediction[i])
