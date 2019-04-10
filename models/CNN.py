import tensorflow as tf
import os

def conv1d_layer(x, filters, kernel_size):
    """This is a 1d conv, so filter_shape = [dim, input_channels, out_channels]"""

    x = tf.layers.conv1d(x, filters, kernel_size)
    x = tf.nn.relu(x)
    return x

def max_pool1d_layer(inp, ksize, strides):
    """tf.nn does not have max_pool_1d, so we have to expand the incoming layer
    as if we were dealing with a 2D convolution and then squeeze it again.
    Again, since this is a 1D conv, the size of the window (ksize) and the stride
    of the sliding window must have only one dimension (height) != 1
    """
    x = tf.expand_dims(inp, 3)
    x = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="VALID")
    x = tf.squeeze(x, [3])
    return x

def batch_norm_layer(inp):
    """As explained in A. Gerón's book, in the default batch_normalization
    there is no scaling, i.e. gamma is set to 1. This makes sense for layers
    with no activation function or ReLU (like ours), since the next layers
    weights can take care of the scaling. In other circumstances, include
    scaling
    """
    # get the size from input tensor (remember, 1D convolution -> input tensor 3D)
    size = int(inp.shape[2])

    batch_mean, batch_var = tf.nn.moments(inp,[0])
    scale = tf.Variable(tf.ones([size]))
    beta  = tf.Variable(tf.zeros([size]))
    x = tf.nn.batch_normalization(inp,batch_mean,batch_var,beta,scale,
        variance_epsilon=1e-3)
    return x


def get_model(X,is_training, filters, W=None, n_classes=23, tf_idf=False, logger=None):
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
        net = tf.nn.embedding_lookup(W, X)
    else:
        net  = X
    #     net = tf.expand_dims(X, axis=-1)  # Change the shape to [batch_size,1,,output_size]
    logger.info("Model representation {}".format(net))
    for i, f in enumerate(filters):
        logger.info("Conv{}".format(i))
        with tf.name_scope("conv{}".format(i)):
            net = conv1d_layer(net, filters=f, kernel_size=5)
            net = max_pool1d_layer(net, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1])
            net = batch_norm_layer(tf.cast(net, dtype=tf.float32))
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