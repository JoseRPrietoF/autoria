import tensorflow as tf
import numpy as np
from models.CNN import *


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 8  # Any size is accepted
num_hidden = 64
DEV_SPLIT = 0.2
NUM_EPOCH = 6
vocab_size = 6000
weights = np.random.randn(vocab_size, EMBEDDING_DIM)
weights = np.asarray(weights, dtype=np.float32)
palNum = {
    "a":0,
    "b":1,
    "c":2,
}



batch_size = tf.placeholder(tf.int64, name="batch_size")
X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int64, shape=[None])

embeddings = tf.Variable(
    tf.random_uniform([2000, EMBEDDING_DIM], -1.0, 1.0))


"""
GET THE MODEL
"""
logits = get_model(X, embeddings)
print(logits)
""""""



