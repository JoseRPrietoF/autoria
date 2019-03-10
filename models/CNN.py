import tensorflow as tf

def conv1d(X, filters=1, stride=1, kernel=(3,1), activation=tf.nn.relu):
    '''
    :param input_: A tensor of embedded tokens with shape [batch_size,max_length,embedding_size]
    :param output_size: The number of feature maps we'd like to calculate
    :param width: The filter width
    :param stride: The stride
    :return: A tensor of the concolved input with shape [batch_size,max_length,output_size]
    '''
    # inputSize = input_.get_shape()[-1] # How many channels on the input (The size of our embedding for instance)

    #This is the kicker where we make our text an image of height 1
    X = tf.expand_dims(X, axis=-1) #  Change the shape to [batch_size,1,max_length,output_size]

    X = tf.cast(X, dtype=tf.float32)
    #Make sure the height of the filter is 1
    convolved = tf.layers.conv2d(
        X,
        filters, kernel,
        activation=activation,
        strides=[stride,1],
        padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
        use_bias=True,
    )
    print("Convolved: {}".format(convolved))
    #Remove the extra dimension, eg make the shape [batch_size,max_length,output_size]
    result = tf.squeeze(convolved, axis=-2)
    return result

def get_model(X, W, n_classes=23):
    tf_word_representation_layer = tf.nn.embedding_lookup(W, X
    )
    X = tf.expand_dims(X, axis=-1)  # Change the shape to [batch_size,1,max_length,output_size]
    print("Model representation {}".format(tf_word_representation_layer))
    net = conv1d(X, 1, stride=2)
    print(net)
    net = conv1d(net, 1, )
    print(net)
    net = tf.layers.dense(tf.layers.flatten(net), n_classes)

    return net

if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 50
    X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])

    logits = get_model(X)
    print(logits)