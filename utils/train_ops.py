import tensorflow as tf


def train_op(X,y, learning_rate=0.01):
    y = tf.cast(y, dtype=tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=X, labels=y)
    loss = tf.reduce_sum(loss)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.5,
                                   beta2=0.999,
                                   epsilon=1e-8,
                                   )

    return optim.minimize(loss), loss