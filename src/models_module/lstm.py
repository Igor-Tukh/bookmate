import tensorflow as tf
import data_generation as dg


class LSTM:
    def __init__(self, learning_rate, n_hidden, n_input, input_dimension, output_dimension, batch_size):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension  # number of features
        self.output_dimension = output_dimension  # number of labels
        self.n_input = n_input  # number of time periods
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, self.n_input, self.input_dimension], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.output_dimension], name='y')

        # RNN output node weights and biases
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.output_dimension]), name='weigths')
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.output_dimension]), name='biases')
        }
        self.pred, self.loss, self.optimizer = self.set_optimizers()

    def RNN(self):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, 0.8)
        x = tf.unstack(self.x, self.n_input, 1)
        outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def set_optimizers(self):
        pred = self.RNN()
        # Loss and optimizer
        loss = tf.reduce_mean(tf.square(pred - self.y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # Model evaluation
        return pred, loss, optimizer

    def train(self, train_x, train_y, test_x, test_y, epochs):
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            epoch, offset = 0, 0
            while epoch <= epochs:
                x_batch, y_batch = dg.get_batch(train_x, train_y, self.batch_size, offset)
                session.run([self.optimizer], feed_dict={self.x: x_batch, self.y: y_batch})
                offset += self.batch_size + 1

                if offset >= len(train_y):
                    # Calculate batch accuracy
                    pred = session.run([self.pred],
                                        feed_dict={self.x: test_x, self.y: test_y})
                    # Calculate batch loss
                    loss = session.run(self.loss, feed_dict={self.x: test_x, self.y: test_y})
                    print('Epoch = %d, Average Loss = %.6f' % (epoch, loss))
                    epoch += 1
                    offset = 0
            dg.results_to_csv(tf.constant(pred).eval().tolist(), test_y)


def main():
    # input_dimension = len(train_x[0][0])
    # train_x, train_y, test_x, test_y = dg.data_generation_from_features(5, 'page_speed')
    train_x, train_y, test_x, test_y = dg.data_generation_from_one_hot('page_speed')
    lstm = LSTM(learning_rate=0.001, n_hidden=128, input_dimension=len(train_x[0][0]),
                output_dimension=1, n_input=1000, batch_size=16)
    lstm.train(train_x, train_y, test_x, test_y, epochs=10)


if __name__ == "__main__":
    main()