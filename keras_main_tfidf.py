import numpy as np
from data import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
"""
Vars
"""
root_path = "/data2/jose/data_autoria/"
train_path = root_path+"train"
test_path = root_path+"test"
fname_vocab = root_path+"vocabulario"



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
# keras
"""""""""""""""""""""""""""""""""""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# MODEL = "RNN"
# MODEL = "CNN"
MODEL = "FF"
#FOR RNN
HIDDEN_UNITS = 32
layers = [256,128,64,32]
BATCH_SIZE = 64  # Any size is accepted
NUM_EPOCH = 10
n_classes = 4

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(layers[0], activation='relu', input_dim=len(texts_rep_train[0])))
model.add(Dropout(0.5))
for l in layers[1:]:
    model.add(Dense(l, activation='relu'))
    model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', f1])
print(model.summary())
model.fit(texts_rep_train, y_train,
          epochs=NUM_EPOCH,
          batch_size=BATCH_SIZE)
_, acc_score, f1_score = model.evaluate(text_test_rep, y_test, batch_size=BATCH_SIZE)

print("\n\n ------------------ \n\n")
print("Acuracy: {}".format(acc_score))
print("F1: {}".format(f1_score))