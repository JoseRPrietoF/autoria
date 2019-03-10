from data import process
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from models.CNN import *
from utils import train_ops
"""
Vars
"""
root_path = "/data2/jose/data_autoria/"
train_path = root_path+"train"
test_path = root_path+"test"
fname_vocab = root_path+"vocabulario"


dir_word_embeddings = '/data2/jose/word_embedding/glove-sbwc.i25.vec'

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 8  # Any size is accepted
num_hidden = 64
DEV_SPLIT = 0.2
NUM_EPOCH = 6

"""
"""

print("Loading train dataset...")
dt_train = process.canon60Dataset(train_path)
print("Loaded")

print("Loading test dataset...")
dt_test = process.canon60Dataset(test_path)
print("Loaded")

classes = {}

for c in dt_train.y:
    c_count = classes.get(c, 0)
    classes[c] = c_count + 1

n_classes = len(classes.keys())
print("N_Classes: {}".format(n_classes))
[print("{} -> {}".format(x, classes[x])) for x in classes.keys()]
print("-"*30)
vocab_freq = process.read_vocab(fname_vocab)
vocab = list(vocab_freq.keys())
vocab_size = len(vocab)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Init
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Words to numbers
palNum = {}
numPal = {}
classNum = {}
numClass = {}

for i, word in enumerate(vocab):
    palNum[word] = i+1
    numPal[i+1] = word

for i, c in enumerate(list(classes.keys())):
    classNum[c] = i
    numClass[i] = c

X = []
y = []

for i,sentence in enumerate(dt_train.X):
    tempX = []
    c = dt_train.y[i]
    for words in sentence:
        words = words.split()
        for word in words:
            tempX.append(palNum[word])

    X.append(tempX)
    y.append(classNum[c])

#onehot
n_values = np.max(y) + 1
y = np.eye(n_values)[y]

print(len(X))
print(len(y))

embeddings_index = {}
embeddings_tmp = []
with open(dir_word_embeddings, encoding="utf8") as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs[:EMBEDDING_DIM]
        item = palNum.get(word, False)
        if item:
            embeddings_tmp.append(coefs)
        # else:
        #     rand_num = np.random.uniform(low=-0.2, high=0.2, size=EMBEDDING_DIM)
        #     embeddings_tmp.append(rand_num)

# final embedding array corresponds to dictionary of words in the document
embedding = np.asarray(embeddings_tmp)

n_etiquetas = n_classes
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=DEV_SPLIT, random_state=3)

print(len(X_train))
print(X_train[0].shape)

print('Datos de entrenamiento {}'.format(len(X_train)))
print('Datos de validacion {}'.format(len(X_val)))
print('Numero total de vectores {}'.format(len(embeddings_index)))

"""""""""""""""""""""""""""""""""""
# Tensorflow
"""""""""""""""""""""""""""""""""""

batch_size = tf.placeholder(tf.int64, name="batch_size")
X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int64, shape=[None, n_classes])

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
logits = get_model(X, embeddings)
print(logits)
softmax = tf.nn.softmax(logits)

train_op, loss = train_ops.train_op(logits,y)
""""""

train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)
dev_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(buffer_size=12)

train_data = (X_train, y_train)
dev_data = (X_val, y_val)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                       train_dataset.output_shapes)

# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
dev_init_op = iter.make_initializer(dev_dataset)

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
                                  })
        print(loss_result)
        # for i in range(len(batch_tgt)):
        #     print(batch_x[i])
        #     print(batch_tgt[i])
        #     print("--")
        # print("-----------")

    print("Eval")
    ## Eval
    sess.run(dev_init_op, feed_dict={
        X: dev_data[0],
        y: dev_data[1],
        batch_size: BATCH_SIZE,
    }
             )

    current_batch_index = 0
    next_element = iter.get_next()

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
        #
        # for i in range(len(results)):
        #     print(results[i])
        #
        # print("----------")