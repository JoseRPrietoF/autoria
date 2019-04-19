from __future__ import print_function
from __future__ import division
import logging
from utils.optparse import Arguments as arguments
from PAN import tfidf, embeddings, prepare_WE
from data import process

# import embeddings



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

    logger.info("Network model: {}".format(opts.model))
    logger.info("Data representation as : {}".format(opts.represent))
    n_classes = 2
    if opts.represent == "tfidf":
        for l in ['es','en']:

            tfidf.Model(layers=opts.layers, filters=opts.filters, MODEL=opts.model,
                        min_ngram=opts.min_ngram, up=opts.max_ngram,
                        max_features=opts.max_features, NUM_EPOCH=opts.epochs,
                        logger=logger, dataset=opts.dataset, opts=opts, DEBUG=opts.debug, lang=l)
    elif opts.represent == "WE":
        lang = "es"
        path = opts.tr_data + '/' + lang
        path_test = opts.i + '/' + lang

        txt_train = opts.tr_data + "/{}/truth-train.txt".format(lang)
        txt_test = opts.file_i + "/{}/truth-dev.txt".format(lang)
        txt_dev = opts.tr_data + "/{}/truth-dev.txt".format(lang)
        dt_train = process.PAN2019(path=path, txt=txt_train, join_all=False)
        dt_test = process.PAN2019(path=path_test, txt=txt_test, join_all=False)
        dt_dev = process.PAN2019(path=path, txt=txt_dev, join_all=False)


        x_train = dt_train.X
        y_train = dt_train.y
        fnames_train = dt_train.fnames

        x_dev = dt_dev.X
        y_dev = dt_dev.y
        fnames_dev = dt_dev.fnames

        x_test = dt_test.X
        y_test = dt_test.y
        fnames_test = dt_test.fnames

        vocab, x_train_tokenized = dt_train.get_vocab()


        WE_path = '/data2/jose/word_embedding/glove-sbwc.i25.vec'
        EMBEDDING_DIM = 300
        MAX_SEQUENCE_LENGTH = 50
        X_train, X_test, X_dev, embedding, MAX_SEQUENCE_LENGTH = \
            prepare_WE.prepare_data(WE_path, x_train, x_test,
                                    vocab, EMBEDDING_DIM, logger=logger, opts=opts, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, x_dev=x_dev)

        model = embeddings.Model(X_train, y_train, X_test, y_test, fnames_train, fnames_test,
                                 X_dev, y_dev, fnames_dev,
                                 embedding,
                                 MAX_SEQUENCE_LENGTH,
                                 EMBEDDING_DIM=300,
                                 OPTIMIZER='adam',
                                 logger=logger,
                                 opts=opts,
                                 by_acc=True,
                                 n_classes=2, )
    else:
        logger.info("Data representation {} is not recognized".format(opts.represent))
        exit()


if __name__ == "__main__":
    main()
