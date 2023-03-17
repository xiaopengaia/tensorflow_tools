import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

class Config:
    def __init__(self):
        self.save_path = '../models/p34/conv_mnist'
        self.sample_path = '../samples/MNIST_data'
        self.lr = 0.001
        self.epoches = 2
        self.batch_size = 200


class Tensors:
    def __init__(self, config: Config):
        self.x = tf.placeholder(tf.float32, [None, 784], 'x')
        self.y = tf.placeholder(tf.int32, [None], 'y')

        x = tf.reshape(self.x, [-1, 28, 28, 1])
        x = tf.layers.conv2d(x, 16, 3, 1, 'same', activation=tf.nn.relu)   # [-1, 28, 28, 16]

        x = tf.layers.conv2d(x, 32, 3, 2, 'same', activation=tf.nn.relu)   # [-1, 14, 14, 32]
        x = tf.layers.conv2d(x, 64, 3, 2, 'same', activation=tf.nn.relu)   # [-1, 7, 7, 64]

        x = tf.layers.flatten(x)   # [-1, 7*7*64]
        logits = tf.layers.dense(x, 10)

        y = tf.one_hot(self.y, 10)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        self.loss = tf.reduce_mean(self.loss)

        self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)  # [-1]
        self.precise = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
        self.precise = tf.reduce_mean(self.precise)

        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)


class App:
    def __init__(self, config: Config):
        self.config = config
        self.ts = Tensors(config)
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.session, config.save_path)
            print('Restore model from %s successfully!' % config.save_path)
        except:
            print('Fail to restore the model from %s, use a new empty one.' % config.save_path)
            self.session.run(tf.global_variables_initializer())

    def train(self, ds):
        cfg = self.config
        ts = self.ts

        batches = ds.train.num_examples // cfg.batch_size
        for epoch in range(cfg.epoches):
            # MBGD
            for batch in range(batches):
                xs, ys = ds.train.next_batch(cfg.batch_size)
                _, loss = self.session.run([ts.train_op, ts.loss], {ts.x: xs, ts.y: ys, ts.lr: cfg.lr})
                print('%d/%03d. loss = %.6f' % (epoch, batch, loss))
            xs, ys = ds.validation.next_batch(cfg.batch_size)
            precise = self.session.run(ts.precise, {ts.x: xs, ts.y: ys})
            print('%d: precise: %.6f' % (epoch, precise), flush=True)
            self.save()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path)

    def predict(self, ds, batch_size=200):
        precise_total = 0
        batches = ds.test.num_examples // batch_size
        for _ in range(batches):
            xs, ys = ds.test.next_batch(batch_size)
            precise = self.session.run(self.ts.precise, {self.ts.x: xs, self.ts.y: ys})
            precise_total += precise
        print('Precise:', precise_total / batches)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    cfg = Config()
    app = App(cfg)
    ds = read_data_sets(cfg.sample_path)
    with app:
        app.train(ds)
        app.predict(ds)
