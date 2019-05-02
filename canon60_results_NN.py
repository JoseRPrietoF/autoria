from __future__ import print_function
from __future__ import division
import logging, os
import numpy as np
from utils.optparse import Arguments as arguments
from canon60 import tfidf
from data import process
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

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



def evaluate(y_test, pred):
    acc = accuracy_score(y_test, pred)
    f1_micro = f1_score(y_test, pred, average='micro')
    p_micro = precision_score(y_test, pred, average='micro')
    r_micro = recall_score(y_test, pred, average='micro')

    f1_macro = f1_score(y_test, pred, average='macro')
    p_macro = precision_score(y_test, pred, average='macro')
    r_macro = recall_score(y_test, pred, average='macro')

    return acc, f1_macro, p_macro, r_macro


def main():
    """
    Starts all
    :return:
    """
    logger, opts = prepare()
    logger.info("---- CANON60 ----")
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
    logger.info("Results")
    modelos = ["FF", ]
    representation = ["tfidf", ]
    max_features = [500,1000,2000,5000,10000,15000, 20000,len(vocab)]
    # max_features = [500,10000]
    min_ngrams = [1]
    max_ngram = [2,3,4,5,6,7,8,9]
    # max_ngram = [3,4]


    for model_type in modelos:
        for repren in representation:
            model_type_name = model_type+"_"+repren
            logger.info("Clasificador: {}".format(model_type_name))
            file = open("{}/{}".format(opts.work_dir, model_type_name), "w")
            file.write("classifier,max_features,min_ngram,max_ngram,accuracy,f1_macro,precision_macro,recall_macro\n")
            opts.model = model_type
            for max_feat in max_features:

                for min_ngram in min_ngrams:
                    for up in max_ngram:
                        rep = TfidfVectorizer(ngram_range=(min_ngram, up), max_features=max_feat, vocabulary=vocab[:max_feat])
                        texts_rep_train = rep.fit_transform(x_train)
                        texts_rep_train = texts_rep_train.toarray()
                        text_test_rep = rep.transform(x_test)
                        text_test_rep = text_test_rep.toarray()
                        logger.info(texts_rep_train.shape)
                        logger.info(text_test_rep.shape)

                        model = tfidf.Model(texts_rep_train, y_train, text_test_rep, y_test,fnames_train,fnames_test,
                                            layers=opts.layers,
                                    logger=logger, opts=opts)

                        pred, y_true = model.get_results()
                        # pred, y_true = results[0], results[2]

                        acc, f1_macro, p_macro, r_macro = evaluate(y_true, pred)
                        # acc, f1_macro, p_macro, r_macro = random.random(), random.random(), random.random(), random.random()
                        res = "{},{},{},{},{},{},{},{}\n".format(model_type, max_feat, min_ngram, up, acc, f1_macro, p_macro, r_macro)

                        logger.info(res)
                        file.write(res)
                        file.flush()
            file.close()

    logger.info("---- FIN ----")


if __name__ == "__main__":
    main()