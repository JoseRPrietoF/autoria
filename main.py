from __future__ import print_function
from __future__ import division
import logging
from utils.optparse import Arguments as arguments
from PAN import tfidf


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

    if opts.represent == "tfidf":
        for l in ['es','en']:

            tfidf.Model(layers=opts.layers, filters=opts.filters, MODEL=opts.model,
                        min_ngram=opts.min_ngram, up=opts.max_ngram,
                        max_features=opts.max_features, NUM_EPOCH=opts.epochs,
                        logger=logger, dataset=opts.dataset, opts=opts, DEBUG=opts.debug, lang=l)
    elif opts.represent == "WE":
        logger.info("WE not implemented by the moment")
        exit()
    else:
        logger.info("Data representation {} is not recognized".format(opts.represent))
        exit()


if __name__ == "__main__":
    main()
