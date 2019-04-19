from data import process
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



def prepare_data(WE_path,x_train, x_test, vocab, EMBEDDING_DIM, opts, logger, MAX_SEQUENCE_LENGTH=None, x_dev=None):
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
    DEBUG_ = opts.debug
    vocab_size = len(vocab)

    # Words to numbers
    palNum = {}
    numPal = {}

    for i, word in enumerate(vocab):
        palNum[word] = i
        numPal[i] = word

    X = []
    X_test = []
    X_dev = []

    for i, sentence in enumerate(x_train):
        tempX = []
        for word in sentence.split():
            num = palNum.get(word, False)
            if num:
                tempX.append(palNum[word])

        X.append(tempX)

    for i, sentence in enumerate(x_test):
        tempX = []
        for word in sentence.split():

            num = palNum.get(word, False)
            if num:
                #TODO add unknown words
                tempX.append(num)

        X_test.append(tempX)

    if x_dev is not None:
        for i, sentence in enumerate(x_dev):
            tempX = []
            for word in sentence.split():

                num = palNum.get(word, False)
                if num:
                    # TODO add unknown words
                    tempX.append(num)

            X_dev.append(tempX)

    logger.info("Total data to train: {}".format(len(X)))
    if MAX_SEQUENCE_LENGTH is None:
        MAX_SEQUENCE_LENGTH = max(len(l) for l in X)
        logger.info("Setting MAX_SEQUENCE_LENGTH to {} automatically".format(MAX_SEQUENCE_LENGTH))

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



    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    if x_dev is not None:
        X_dev = pad_sequences(X_dev, maxlen=MAX_SEQUENCE_LENGTH)

    print(len(X))
    print(X_test[0].shape)

    logger.info('Datos de entrenamiento {}'.format(X.shape))
    logger.info('Datos de test {}'.format(X_test.shape))
    if x_dev is not None:
        logger.info('Datos de val {}'.format(X_dev.shape))
    if not DEBUG_:
        logger.info('Numero total de vectores {}'.format(len(embeddings_index)))

    if x_dev is not None:
        return X, X_test, X_dev, embedding, MAX_SEQUENCE_LENGTH
    else:
        return X, X_test, embedding, MAX_SEQUENCE_LENGTH
