import tensorflow as tf


def conv2d(net, filters, kernel, strides):
    # net = tf.layers.conv2d(
    #     net,
    #     filters, kernel,
    #     activation=None,
    #     strides=strides,
    #     padding='same',
    #     kernel_initializer=tf.keras.initializers.he_normal(),
    #     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
    #     use_bias=False,
    # )

    net = tf.layers.separable_conv2d(
        net,
        filters, kernel,
        activation=None,
        strides=strides,
        padding='same',
        # kernel_initializer=tf.keras.initializers.he_normal(),
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
        use_bias=False,
    )
    return net


def get_model(X,is_training, filters,
              num_tweets=100,
              kernel_size=3,
              # kernel_size=300,
              W=None, n_classes=2, tf_idf=False, logger=None,
              opts=None,
                n_gramas=2
              ):
    """
    doc here :)
    :param X:
    :param W:
    :param is_training:
    :param n_classes:
    :return:
    """
    logger.info("CREATING THE MODEL \n")
    tf.logging.set_verbosity(tf.logging.FATAL)
    logger.info(X)
    if not tf_idf:
        net = tf.nn.embedding_lookup(W, X) # 50*100 = 5000 - [BATCH, 300, 5000]
        split = tf.split(net, num_tweets, axis=1)
        net = tf.stack(split, axis=-1) # [BATCH, 300, 5000] -> [BATCH, 50, 300, 100]
        # net = tf.transpose(net, [0,2,1])
    else:
        net  = X
        # net = tf.expand_dims(X, axis=-1)  # Change the shape to [batch_size,1,,output_size]
    logger.info("Model representation {}".format(net))

    for i, f in enumerate(filters):
        logger.info("Conv{}".format(i))
        with tf.name_scope("conv{}".format(i)):
            net = conv2d(net, f, kernel=(n_gramas, kernel_size), strides=(1,1))
            # net = conv2d(net, f, kernel=(50, n_gramas), strides=(1,1))
            # net = conv2d(net, f, kernel=(3, 3), strides=(1,1))
            print(net)
            net = tf.nn.max_pool(net, ksize=(1,2,1,1), strides=(1,2,1,1), padding="SAME")
            logger.info(net)

    net = tf.layers.flatten(net)
    logger.info("Flatten {}".format(net))
    net = tf.layers.dropout(net, 0.5, training=is_training)
    net = tf.layers.dense(net, 32)
    logger.info("First dense {}".format(net))
    net = tf.layers.dense(net, n_classes)

    return net

if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 50
    X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])

    logits = get_model(X)
    print(logits)