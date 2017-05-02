import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# bitrate samples
x = tf.placeholder(tf.float32, [None, 3], 'x')
# weights
W = tf.Variable(tf.zeros([3, 1]), name='W')
# biases
b = tf.Variable(tf.zeros([1]), name='b')
# output
y = tf.matmul(x, W) + b

# -- training --
# label data
y_ = tf.placeholder(tf.float32, [None, 1], 'y_')
cross_entropy = tf.reduce_mean(tf.squared_difference(y_, y))
# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)
# tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0002).minimize(cross_entropy)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# visualization
tf.summary.tensor_summary('y', y)
tf.summary.tensor_summary('W', W)
tf.summary.histogram('W histogram', W)
tf.summary.tensor_summary('b', b)
tf.summary.histogram('b histogram', b)
tf.summary.tensor_summary('error', cross_entropy)
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./linear.log', session.graph)

for i in range(10000):
    summary, c_e, y0, w0, b0, _ = session.run([merged, cross_entropy, y, W, b, train_step], feed_dict={
        x: [[1, 2, 3],
             [4, 5, 6],
             [22, 23, 24]],
        y_: [[4], [7], [25]]
    })
    summary_writer.add_summary(summary, i)

# -- eval --
correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(y, feed_dict={
    x: [[1, 2, 3],
        [4, 5, 6],
        [22, 23, 24],
        [78, 79, 80],
        [101, 111, 121]],
    y_: [[4], [7], [25], [81], [131]]
    # x: [[11, 12, 13],
    #     [19, 20, 21],
    #     [-1, 0, 1]],
    # y_: [[14], [22], [2]]
}))

