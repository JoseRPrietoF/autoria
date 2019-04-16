from __future__ import print_function
from __future__ import division
import logging, os
import numpy as np
from utils.optparse import Arguments as arguments
from canon60 import tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from data import process
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV
import random
import utils.utils as utils
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


def train_model(X_train, y_train, classifier="SVC"):
    clf = None
    if classifier == "SVC":
        tuned_parameters = [{'kernel': ['rbf', 'linear'],
        # 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100]},]
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3],
        #                      'C': [1]}, ]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=3, scoring='f1_macro')
    elif classifier == "KNN":
        tuned_parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [3,4,5],
                             'algorithm': ['auto', 'ball_tree', 'kd_tree']}, ]
        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=3, scoring='f1_macro')
    elif classifier == "GaussianNB":
        tuned_parameters = [{'var_smoothing': [1e-9, 1e-8]}, ]
        clf = GridSearchCV(GaussianNB(), tuned_parameters, cv=3, scoring='f1_macro')
    elif classifier == "MultinomialNB":
        tuned_parameters = [{'alpha': [0,1.0,0.5,1.5], 'fit_prior':[True,False]}, ]
        clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=3, scoring='f1_macro')
    elif classifier == "LogisticRegression":
        tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [1.0,0.5,2.0,10.0]}, ]
        clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=3, scoring='f1_macro')
    elif classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier()
    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    return clf

def evaluate(clf, X_test, y_test):
    pred = clf.predict(X_test)
    # cm = confusion_matrix(y_test, pred)

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

    x_test = dt_test.X
    y_test = dt_test.y
    vocab = process.read_vocab_list(opts.vocab_path)
    logger.info("Results")
    # clasificadores = ["GaussianNB", "MultinomialNB", "LogisticRegression", "DecisionTreeClassifier",  "RandomForestClassifier", "SVC", "KNN", ]
    clasificadores = ["LogisticRegression", "DecisionTreeClassifier",  "RandomForestClassifier", "SVC", "KNN", ]
    max_features = [500,1000,2000,4000,8000,10000,20000,len(vocab)]
    # max_features = [10000]
    min_ngrams = [1]
    max_ngram = [2,3,4,5,6,7,8,9]
    # max_ngram = [3]


    for clasif in clasificadores:
        logger.info("Clasificador: {}".format(clasif))
        file = open("{}/{}".format(opts.work_dir, clasif), "w")
        file.write("classifier,max_features,min_ngram,max_ngram,accuracy,f1_macro,precision_macro,recall_macro\n")
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

                    clf = train_model(texts_rep_train, y_train, classifier=clasif)
                    acc, f1_macro, p_macro, r_macro = evaluate(clf, text_test_rep, y_test)
                    # acc, f1_macro, p_macro, r_macro = random.random(), random.random(), random.random(), random.random()
                    res = "{},{},{},{},{},{},{},{}\n".format(clasif, max_feat, min_ngram, up, acc, f1_macro, p_macro, r_macro)
                    logger.info(res)
                    file.write(res)
        file.close()

    logger.info("---- FIN ----")


if __name__ == "__main__":
    main()