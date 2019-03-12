import numpy as np
import tensorflow as tf

from models import CNN
from data.prepare_text import prepare_data
from utils import train_ops
from utils import metrics
"""
Vars
"""
root_path = "/data2/jose/data_autoria/"
train_path = root_path+"train"
test_path = root_path+"test"
fname_vocab = root_path+"vocabulario"


dir_word_embeddings = '/data2/jose/word_embedding/glove-sbwc.i25.vec'
OPTIMIZER = 'rms'
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = None
BATCH_SIZE = 16  # Any size is accepted
DEV_SPLIT = 0.2
NUM_EPOCH = 1000
filters = [64,128,256,512]

X_train, X_val, X_test, y_train, y_val, y_test, embedding, n_classes, MAX_SEQUENCE_LENGTH = prepare_data(
    dir_word_embeddings, fname_vocab, train_path, test_path, EMBEDDING_DIM,
    VALIDATION_SPLIT=DEV_SPLIT, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH
)
"""""""""""""""""""""""""""""""""""
# Tensorflow
"""""""""""""""""""""""""""""""""""

batch_size = tf.placeholder(tf.int64, name="batch_size")
X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int64, shape=[None, n_classes])
is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

glove_weights_initializer = tf.constant_initializer(embedding)
print(glove_weights_initializer)
embeddings = tf.Variable(
    # tf.random_uniform([2000, EMBEDDING_DIM], -1.0, 1.0)
    embedding
)

print(embeddings)
"""
GET THE MODEL
"""
logits = CNN.get_model(X, embeddings, is_training, filters=filters, n_classes=n_classes)
print(logits)
softmax = tf.nn.softmax(logits)

num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

print("{} params to train".format(num_params))

train_op, loss = train_ops.train_op(logits,y, optimizer=OPTIMIZER)
""""""
"""Test de embeddings"""

train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)
dev_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)
test_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)

train_data = (X_train, y_train)
dev_data = (X_val, y_val)
test_data = (X_test, y_test)


# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                       train_dataset.output_shapes)

# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
dev_init_op = iter.make_initializer(dev_dataset)
test_init_op = iter.make_initializer(test_dataset)

epoch_start = 0
## Train
sess = tf.Session()
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_g)
sess.run(init_l)
for epoch in range(epoch_start, NUM_EPOCH+1):
    sess.run(train_init_op, feed_dict={
            X: train_data[0],
            y: train_data[1],
            batch_size: BATCH_SIZE,
        }
    )

    current_batch_index = 0
    next_element = iter.get_next()
    loss_count = 0
    while True:

        try:
            data = sess.run([next_element])
        except tf.errors.OutOfRangeError:
            break

        current_batch_index += 1
        data = data[0]
        batch_x, batch_tgt = data

        _, loss_result = sess.run([train_op, loss],
                                  feed_dict={
                                      X: batch_x,
                                      y: batch_tgt,
                                      batch_size: BATCH_SIZE,
                                      is_training: True
                                  })
        loss_count += loss_result

    loss_count = loss_count / current_batch_index
    print("Loss on epoch {} : {}".format(epoch, loss_count))
    print("Eval")
    ## Eval
    sess.run(dev_init_op, feed_dict={
    # sess.run(dev_init_op, feed_dict={
        X: dev_data[0],
        y: dev_data[1],
        batch_size: BATCH_SIZE,
    }
             )

    current_batch_index = 0
    next_element = iter.get_next()

    acc = 0
    while True:

        try:
            data = sess.run([next_element])
        except tf.errors.OutOfRangeError:
            break

        current_batch_index += 1
        data = data[0]
        batch_x, batch_tgt = data

        results = sess.run([softmax],
                                  feed_dict={
                                      X: batch_x,
                                      y: batch_tgt,
                                      batch_size: BATCH_SIZE,
                                  })

        for i in range(len(results)):
            acc_aux = metrics.accuracy(X=results[i], y=batch_tgt[i])
            acc += acc_aux

    acc = acc / current_batch_index
    print("Acc Val epoch {} : {}".format(epoch, acc))
    print("----------")

"""
----------------- TEST -----------------
"""
print("\n-- TEST --\n")

sess.run(test_init_op, feed_dict={
    X: test_data[0],
    y: test_data[1],
    batch_size: BATCH_SIZE,
}
         )

current_batch_index = 0
next_element = iter.get_next()
loss_count = 0
while True:

    try:
        data = sess.run([next_element])
    except tf.errors.OutOfRangeError:
        break

    current_batch_index += 1
    data = data[0]
    batch_x, batch_tgt = data

    results = sess.run([softmax],
                       feed_dict={
                           X: batch_x,
                           y: batch_tgt,
                           batch_size: BATCH_SIZE,
                       })

    for i in range(len(results)):
        acc_aux = metrics.accuracy(X=results[i], y=batch_tgt[i])
        acc += acc_aux

acc = acc / current_batch_index
print("----------")
print("Acc Val Test {}".format(acc))
print("----------")