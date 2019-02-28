import numpy as np
import tensorflow as tf


input_size = 2
hidden_size = 3
out_size = 1


def generate_test_data():
    inp = 0.5*np.random.rand(10, 2)
    oup = np.zeros((10, 1))
    for idx, val in enumerate(inp):
        oup[idx] = np.array([val[0] + val[1]])
    return inp, oup


def create_network():
    x = tf.placeholder(tf.float32, [None, input_size])

    w01 = tf.Variable(tf.truncated_normal([input_size, hidden_size], stddev=0.1))
    y1 = tf.matmul(x, w01)

    w12 = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.1))
    y2 = tf.matmul(y1, w12)

    y_ = tf.placeholder(tf.float32, [None, out_size])
    return x, y_, y2


def train(x, y_, y2):
    cross_entropy = tf.reduce_mean(
        tf.nn.l2_loss(y_ - y2)
    )
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for i in range(100000):
        batch_xs, batch_ys = generate_test_data()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test
        if i % 2000 == 0:
            out_batch = sess.run(y2, {x: batch_xs})
            inx = 0
            print(batch_xs[inx][0], " + ", batch_xs[inx][1], " = ", out_batch[inx][0], "|", batch_xs[inx][0] + batch_xs[inx][1])


(x, y_, y2) = create_network()
train(x, y_, y2)