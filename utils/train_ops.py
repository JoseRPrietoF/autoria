import tensorflow as tf


def train_op(X,y, learning_rate=0.01):
    y = tf.cast(y, dtype=tf.float32)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=X)
    loss = tf.reduce_sum(loss)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8,
                                   )

    return optim.minimize(loss), loss