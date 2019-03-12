import tensorflow as tf

def conv1d_layer(x, filter_shape):
    """This is a 1d conv, so filter_shape = [dim, input_channels, out_channels]"""
    # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01))
    # b = tf.Variable(tf.random_normal(shape=[filter_shape[2]]))
    # x = tf.nn.conv1d(inp,W,stride=1,padding="VALID")
    # x = tf.nn.bias_add(x, b)
    x = tf.layers.conv1d(x, filter_shape[2], filter_shape[0])
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


def get_model(X, W,is_training, n_classes=23, EMBEDDING_DIM=300):
    tf_word_representation_layer = tf.nn.embedding_lookup(W, X)
    # X = tf.expand_dims(X, axis=-1)  # Change the shape to [batch_size,1,max_length,output_size]
    print("Model representation {}".format(tf_word_representation_layer))
    print("Conv1")
    with tf.name_scope("conv1"):
        net = conv1d_layer(tf_word_representation_layer, filter_shape=[5, EMBEDDING_DIM, 128])
        print(net)
        net = max_pool1d_layer(net, ksize=[1, 5, 1, 1], strides=[1, 2, 1, 1])
        print(net)
        net = batch_norm_layer(tf.cast(net, dtype=tf.float32))
        print(net)
    print("Conv2")
    with tf.name_scope("conv2"):
        net = conv1d_layer(net, filter_shape=[5, 128, 128])
        print(net)
        net = max_pool1d_layer(net, ksize=[1, 5, 1, 1], strides=[1, 2, 1, 1])
        print(net)
        net = batch_norm_layer(tf.cast(net, dtype=tf.float32))
        print(net)
    print("Conv3")
    with tf.name_scope("conv3"):
        net = conv1d_layer(net, filter_shape=[5, 128, 64])
        print(net)
        net = max_pool1d_layer(net, ksize=[1, 5, 1, 1], strides=[1, 2, 1, 1])
        print(net)
        net = batch_norm_layer(tf.cast(net, dtype=tf.float32))
        print(net)
    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, 0.5, training=is_training)
    net = tf.layers.dense(net, 128)
    net = tf.layers.dense(net, n_classes)

    return net

if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 50
    X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])

    logits = get_model(X)
    print(logits)