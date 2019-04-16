from __future__ import print_function
from __future__ import division
import logging, os
import numpy as np
from utils.optparse import Arguments as arguments
from canon60 import tfidf, prepare_WE, embeddings
# import embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from data import process


def prepare():
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()

    fh = logging.FileHandler(opts.log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)

    return logger, opts


def main():
    """
    Starts all
    :return:
    """
    logger, opts = prepare()
    logger.info("---- CANON60 ----")
    logger.info("Network model: {}".format(opts.model))
    logger.info("Data representation as : {}".format(opts.represent))
    train_path = opts.i + "/train"
    test_path = opts.i + "/test"
    dt_train = process.canon60Dataset(train_path, join_all=True)
    dt_test = process.canon60Dataset(test_path, join_all=True)

    x_train = dt_train.X
    y_train = dt_train.y
    fnames_train = dt_train.fnames

    x_test = dt_test.X
    y_test = dt_test.y
    fnames_test = dt_test.fnames
    vocab = process.read_vocab_list(opts.vocab_path)
    if opts.represent == "tfidf":
        min_ngram, up = 1,7
        max_feat = 23000
        rep = TfidfVectorizer(ngram_range=(min_ngram, up), max_features=max_feat, vocabulary=vocab[:max_feat])
        texts_rep_train = rep.fit_transform(x_train)
        texts_rep_train = texts_rep_train.toarray()
        text_test_rep = rep.transform(x_test)
        text_test_rep = text_test_rep.toarray()
        logger.info(texts_rep_train.shape)
        logger.info(text_test_rep.shape)
        tfidf.Model(texts_rep_train, y_train, text_test_rep, y_test,fnames_train,fnames_test,
                    layers=opts.layers,
                    logger=logger, opts=opts)
    elif opts.represent == "WE":

        WE_path = '/data2/jose/word_embedding/glove-sbwc.i25.vec'
        EMBEDDING_DIM = 300
        X_train, X_test, embedding, MAX_SEQUENCE_LENGTH = \
            prepare_WE.prepare_data(WE_path, x_train, x_test,
                                    vocab, EMBEDDING_DIM, opts=opts, MAX_SEQUENCE_LENGTH=None)

        model = embeddings.Model(X_train, y_train, X_test, y_test,fnames_train,fnames_test,
                 embedding,
                 MAX_SEQUENCE_LENGTH,
                 EMBEDDING_DIM=300,
                 OPTIMIZER='adam',
                 logger=logger,
                 opts=opts,
                 by_acc=True,
                 n_classes = 4,)
    else:
        logger.info("Data representation {} is not recognized".format(opts.represent))
        exit()


if __name__ == "__main__":
    main()
