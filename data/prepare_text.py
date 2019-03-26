from data import process
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

DEBUG_ = False

def prepare_data(WE_path,FNAME_VOCAB,TEXT_DATA_DIR_TRAIN, TEXT_DATA_DIR_TEST,EMBEDDING_DIM,VALIDATION_SPLIT=0.2, MAX_SEQUENCE_LENGTH=None,
                 reset_test=False):
    """
    Preparing the data and the word embeddings to the model
    :param WE_path:
    :param FNAME_VOCAB:
    :param TEXT_DATA_DIR_TRAIN:
    :param TEXT_DATA_DIR_TEST:
    :param MAX_SEQUENCE_LENGTH: Max seq. length to cut or add padding. If its None then MAX = longest length of the string
    :param EMBEDDING_DIM:
    :param VALIDATION_SPLIT:
    :param reset_test: Merge test and train and re-split the test-train partitions
    :return:
    """

    print("Loading train dataset...")
    dt_train = process.canon60Dataset(TEXT_DATA_DIR_TRAIN)
    print("Loaded")

    print("Loading test dataset...")
    dt_test = process.canon60Dataset(TEXT_DATA_DIR_TEST)
    print("Loaded")

    classes = {}

    for c in dt_train.y:
        c_count = classes.get(c, 0)
        classes[c] = c_count + 1

    n_classes = len(classes.keys())
    print("N_Classes: {}".format(n_classes))
    [print("{} -> {}".format(x, classes[x])) for x in classes.keys()]
    print("-" * 30)
    vocab_freq = process.read_vocab(FNAME_VOCAB)
    vocab = list(vocab_freq.keys())
    vocab_size = len(vocab)

    # Words to numbers
    palNum = {}
    numPal = {}
    classNum = {}
    numClass = {}

    for i, word in enumerate(vocab):
        palNum[word] = i
        numPal[i] = word

    for i, c in enumerate(list(classes.keys())):
        classNum[c] = i
        numClass[i] = c

    X = []
    y = []
    X_test, y_test = [],[]

    for i, sentence in enumerate(dt_train.X):
        tempX = []
        c = dt_train.y[i]
        for words in sentence:
            words = words.split()
            for word in words:
                tempX.append(palNum[word])

        X.append(tempX)
        y.append(classNum[c])

    for i, sentence in enumerate(dt_test.X):
        tempX = []
        c = dt_test.y[i]
        for words in sentence:
            words = words.split()
            for word in words:
                num = palNum.get(word, False)
                if num:
                    #TODO add unknown words
                    tempX.append(num)

        X_test.append(tempX)
        y_test.append(classNum[c])

    n_values = np.max(y) + 1
    if reset_test:
        y.extend(y_test)
    else:
        y_test = np.eye(n_values)[y_test]
    # onehot
    y = np.eye(n_values)[y]


    print("Total data to train: {}".format(len(X)))
    if MAX_SEQUENCE_LENGTH is None:
        MAX_SEQUENCE_LENGTH = max(len(l) for l in X)
        print("Setting MAX_SEQUENCE_LENGTH to {} automatically".format(MAX_SEQUENCE_LENGTH))

    embedding = np.random.randn(vocab_size, EMBEDDING_DIM)

    if not DEBUG_:
        embeddings_index = {}

        with open(WE_path, encoding="utf8") as glove_file:

            for line in glove_file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs[:EMBEDDING_DIM]
                item = palNum.get(word, False)

        for word, i in palNum.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding[i] = embedding_vector
    else:
        print("Careful, DEBUG MODE ON")



    if reset_test:
        print("Restoring test partition...")
        X.extend(X_test)
        # y.append(y_test)
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=3)

    print(len(X_train))
    print(X_train[0].shape)

    print('Datos de entrenamiento {}'.format(len(X_train)))
    print('Datos de validacion {}'.format(len(X_val)))
    if not DEBUG_:
        print('Numero total de vectores {}'.format(len(embeddings_index)))
    print("Un total de {} clases".format(n_classes))

    return X_train, X_val, X_test, y_train, y_val, y_test, embedding, n_classes, MAX_SEQUENCE_LENGTH
