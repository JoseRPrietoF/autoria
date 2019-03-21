import numpy as np
import tensorflow as tf
from data import process
from models import CNN
from models import FF
from models import RNN
from data.prepare_text import prepare_data
from utils import train_ops
from utils import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Vars
"""
root_path = "/data2/jose/data_autoria/"
train_path = root_path+"train"
test_path = root_path+"test"
fname_vocab = root_path+"vocabulario"

# MODEL = "RNN"
# MODEL = "CNN"
MODEL = "FF"
#FOR RNN
HIDDEN_UNITS = 32
NUM_LAYERS = 2
do_val = False


OPTIMIZER = 'adam'
BATCH_SIZE = 64  # Any size is accepted
DEV_SPLIT = 0.2
NUM_EPOCH = 100
layers = [64,32]
# filters = [32,64,128,256]
filters = [64,128,256,512]
n_classes = 4

#### DATOS
dt_train = process.canon60Dataset(train_path, join_all=True)
dt_test = process.canon60Dataset(test_path, join_all=True)

x_train = dt_train.X
y_train = dt_train.y


x_test = dt_test.X
y_test = dt_test.y


# onehot
classes = {}

for c in dt_train.y:
    c_count = classes.get(c, 0)
    classes[c] = c_count + 1
classNum = {}
numClass = {}
for i, c in enumerate(list(classes.keys())):
    classNum[c] = i
    numClass[i] = c

y_train_, y_test_ = [],[]

for i, sentence in enumerate(y_test):
    c = dt_test.y[i]
    y_test_.append(classNum[c])

for i, sentence in enumerate(y_train):
    c = dt_train.y[i]
    y_train_.append(classNum[c])

# To One hot
n_values = np.max(y_train_) + 1
y_test = np.eye(n_values)[y_test_]
y_train = np.eye(n_values)[y_train_]



rep=TfidfVectorizer()
texts_rep_train = rep.fit_transform(x_train)
texts_rep_train = texts_rep_train.toarray()

text_test_rep = rep.transform(x_test)
text_test_rep = text_test_rep.toarray()

del dt_train
del dt_test


# X_train, X_val, X_test, y_train, y_val, y_test, MAX_SEQUENCE_LENGTH = prepare_data(
#     dir_word_embeddings, fname_vocab, train_path, test_path, EMBEDDING_DIM,
#     VALIDATION_SPLIT=DEV_SPLIT, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH
# )
"""""""""""""""""""""""""""""""""""
# Tensorflow
"""""""""""""""""""""""""""""""""""

batch_size = tf.placeholder(tf.int64, name="batch_size")
X = tf.placeholder(tf.float32, shape=[None, len(texts_rep_train[0])])
print(X)
y = tf.placeholder(tf.int64, shape=[None, n_classes])
lr = tf.placeholder(tf.float32, shape=[])
is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

"""
GET THE MODEL
"""
if MODEL == "CNN":
    logits = CNN.get_model(X,  is_training= is_training, filters=filters, n_classes=n_classes, tf_idf=True)
elif MODEL == "RNN":
    logits = RNN.get_model(X, dropout_keep_prob, hidden_size = HIDDEN_UNITS, n_classes=n_classes, num_layers=NUM_LAYERS)
elif MODEL == "FF":
    logits = FF.get_model(X, dropout_keep_prob, is_training=is_training, layers=layers)
print(logits)
softmax = tf.nn.softmax(logits)

num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

print("{} params to train".format(num_params))

train_op, loss = train_ops.train_op(logits,y,learning_rate=lr,  optimizer=OPTIMIZER)
""""""
"""Test de embeddings"""

train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)
dev_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)
test_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)

train_data = (texts_rep_train, y_train)
# dev_data = (X_val, y_val)
test_data = (text_test_rep, y_test)


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
                                      lr: train_ops.lr_scheduler(epoch),
                                      batch_size: BATCH_SIZE,
                                      is_training: True,
                                      dropout_keep_prob: 0.5,

                                  })
        loss_count += loss_result

    loss_count = loss_count / current_batch_index
    print("Loss on epoch {} : {} - LR: {}".format(epoch, loss_count, train_ops.lr_scheduler(epoch)))
    acc = 0
    # if do_val:
    #     print("Eval")
    #     ## Eval
    #     sess.run(dev_init_op, feed_dict={
    #     # sess.run(dev_init_op, feed_dict={
    #         X: dev_data[0],
    #         y: dev_data[1],
    #         batch_size: BATCH_SIZE,
    #     }
    #              )
    #
    #     current_batch_index = 0
    #     next_element = iter.get_next()
    #
    #
    #     while True:
    #
    #         try:
    #             data = sess.run([next_element])
    #         except tf.errors.OutOfRangeError:
    #             break
    #
    #         current_batch_index += 1
    #         data = data[0]
    #         batch_x, batch_tgt = data
    #
    #         results = sess.run([softmax],
    #                                   feed_dict={
    #                                       X: batch_x,
    #                                       y: batch_tgt,
    #                                       batch_size: BATCH_SIZE,
    #                                       dropout_keep_prob: 1.0
    #                                   })
    #
    #         for i in range(len(results)):
    #             acc_aux = metrics.accuracy(X=results[i], y=batch_tgt[i])
    #             acc += acc_aux
    #
    #     acc = acc / current_batch_index
    #     print("Acc Val epoch {} : {}".format(epoch, acc))
    #     print("----------")

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
                           dropout_keep_prob:1.0,
                           lr: train_ops.lr_scheduler(1)
                       })

    for i in range(len(results)):
        acc_aux = metrics.accuracy(X=results[i], y=batch_tgt[i])
        acc += acc_aux

acc = acc / current_batch_index
print("----------")
print("Acc Val Test {}".format(acc))
print("----------")