import tensorflow as tf
from tensorflow.contrib import rnn

def get_model(X, W, dropout_keep_prob, hidden_size = 32, num_layers=2 , n_classes=23):
    print("CREATING THE MODEL \n")
    tf.logging.set_verbosity(tf.logging.FATAL)
    print(X)
    net = tf.nn.embedding_lookup(W, X)
    net = tf.cast(net, tf.float32)
    # X = tf.expand_dims(X, axis=-1)  # Change the shape to [batch_size,1,max_length,output_size]
    print("Model representation {}".format(net))
    x_len = tf.reduce_sum(tf.sign(X), 1)
    print(x_len)
    # Recurrent Neural Network
    with tf.name_scope("birnn"):
        fw_cells = [rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
        bw_cells = [rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
        fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob) for cell in fw_cells]
        bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob) for cell in bw_cells]
        print(fw_cells)
        print(bw_cells)
        rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
            fw_cells, bw_cells, net, sequence_length=x_len, dtype=tf.float32)
        print(rnn_outputs)
        last_output = rnn_outputs[:, -1, :]
        print(last_output)
        # Final scores and predictions
    with tf.name_scope("output"):

        logits = tf.layers.dense(last_output, n_classes)

    return logits

