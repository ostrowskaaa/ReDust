import tensorflow as tf


# https://en.wikipedia.org/wiki/F-score
class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1_score', initializer='zeros')
        self.samples_processed = self.add_weight(name='samples_processed', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) > 3:
            y_true = tf.squeeze(y_true)
        if len(y_pred.shape) > 3:
            y_pred = tf.squeeze(y_pred)

        y_true = tf.round(y_true)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
        fp = tf.reduce_sum(tf.math.multiply((1 - y_true), y_pred))
        fn = tf.reduce_sum(tf.math.multiply(y_true, (1 - y_pred)))

        f1 = tp / (tp + 0.5 * (fp + fn))

        self.f1_score.assign((self.f1_score * self.samples_processed + f1 * tf.cast(tf.shape(y_true)[0], dtype=tf.float32)) / (self.samples_processed + tf.cast(tf.shape(y_true)[0], dtype=tf.float32)))
        self.samples_processed.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        return self.f1_score
