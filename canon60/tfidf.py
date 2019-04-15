import numpy as np
import tensorflow as tf
from data import process
from models import CNN
from models import FF
from models import RNN
from data.prepare_text import prepare_data
from utils import train_ops
from utils import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from utils import sesions
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from canon60.functions import *

class Model:

    def __init__(self,
                 layers=[512, 256, 128, 64, 32],
                 filters=[64, 128, 256, 512],
                 MODEL="FF",
                 HIDDEN_UNITS=32,
                 NUM_LAYERS=2,
                 do_val=False,
                 OPTIMIZER='adam',
                 DEV_SPLIT=0.2,
                 min_ngram=1,
                 up=9,
                 max_features=None,
                 logger=None,
                 opts=None,
                 DEBUG=False,
                 by_acc=True
                 ):

        """
        Vars
        """
        BATCH_SIZE=opts.batch_size
        logger = logger or logging.getLogger(__name__)
        # MODEL = "RNN"
        # MODEL = "CNN"
        if not DEBUG:
            root_path = opts.i
            train_path = root_path + "/train"
            test_path = root_path + "/test"
            fname_vocab = root_path + "/vocabulario"
            n_classes = 4

            #### DATOS
            dt_train = process.canon60Dataset(train_path, join_all= True)
            dt_test = process.canon60Dataset(test_path, join_all= True)

            x_train = dt_train.X
            y_train = dt_train.y

            x_test = dt_test.X
            y_test = dt_test.y

            fnames_train = dt_train.fnames
            fnames_test = dt_test.fnames

            labelencoder = LabelEncoder()  #set
            y_train_ = np.array(y_train).astype(str)
            y_test_ = np.array(y_test).astype(str)
            labelencoder.fit(y_train_)
            y_train_ = labelencoder.transform(y_train_)
            y_test_ = labelencoder.transform(y_test_)
            n_values = len(np.unique(y_train_))
            # To One hot
            y_train = to_categorical(y_train_, n_values)
            y_test = to_categorical(y_test_, n_values)

            if max_features:
                rep = TfidfVectorizer(ngram_range=(min_ngram,up),max_features=max_features)
            else:
                rep = TfidfVectorizer(ngram_range=(min_ngram,up))

            logger.info("Creating tfidf vectors - train")

            texts_rep_train = rep.fit_transform(x_train)
            texts_rep_train = texts_rep_train.toarray()

            logger.info("Creating tfidf vectors - test")

            text_test_rep = rep.transform(x_test)
            text_test_rep = text_test_rep.toarray()

            del dt_train
            del dt_test
            if MODEL == "CNN":
                num = opts.num_tweets
                texts_rep_train = texts_rep_train.reshape(int(texts_rep_train.shape[0]/num), num, texts_rep_train.shape[1])
                text_test_rep = text_test_rep.reshape(int(text_test_rep.shape[0]/num), num, text_test_rep.shape[1])

        else:
            logger.info(" --------------- DEBUG ON ------------------")
            n_classes = 2
            n_vcab = 10000
            train_data = 128
            dev_data = 50
            texts_rep_train = np.random.randn(train_data, 100, n_vcab)
            text_test_rep = np.random.randn(dev_data, 100, n_vcab)
            y_train = np.eye(n_classes)[np.random.choice(n_classes, train_data)]
            y_test = np.eye(n_classes)[np.random.choice(n_classes, dev_data)]

            alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            np_alphabet = np.array(alphabet, dtype="|S1")
            fnames_train = np.random.choice(np_alphabet, [train_data])
            fnames_test = np.random.choice(np_alphabet, [dev_data])
            logger.info("Random data created")
        logger.info("texts_rep_train: {}".format(texts_rep_train.shape))
        logger.info("y_train: {}".format(y_train.shape))



            # X_train, X_val, X_test, y_train, y_val, y_test, MAX_SEQUENCE_LENGTH = prepare_data(
        #     dir_word_embeddings, fname_vocab, train_path, test_path, EMBEDDING_DIM,
        #     VALIDATION_SPLIT=DEV_SPLIT, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH
        # )
        """""""""""""""""""""""""""""""""""
        # Tensorflow
        """""""""""""""""""""""""""""""""""

        batch_size = tf.placeholder(tf.int64, name="batch_size")
        if MODEL == "CNN":
            X = tf.placeholder(tf.float32, shape=[None, texts_rep_train.shape[1], texts_rep_train.shape[2]], name="X")
        else:
            X = tf.placeholder(tf.float32, shape=[None, len(texts_rep_train[0])], name="X")
        print(X)
        y = tf.placeholder(tf.int64, shape=[None, n_classes], name="y")
        fnames_plc = tf.placeholder(tf.string, shape=[None], name="fnames_plc")
        lr = tf.placeholder(tf.float32, shape=[], name="lr")
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout")

        """
        GET THE MODEL
        """
        if MODEL == "CNN":
            logits = CNN.get_model(X, is_training=is_training, filters=filters, n_classes=n_classes, tf_idf=True, logger=logger)
        elif MODEL == "RNN":
            logits = RNN.get_model(X, dropout_keep_prob, hidden_size=HIDDEN_UNITS, n_classes=n_classes,
                                   num_layers=NUM_LAYERS)
        elif MODEL == "FF":
            logits = FF.get_model(X, dropout_keep_prob, is_training=is_training, layers=layers, n_classes=n_classes)
        logger.info(logits)
        softmax = tf.nn.softmax(logits)

        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        logger.info("{} params to train".format(num_params))

        train_op, loss = train_ops.train_op(logits, y, learning_rate=lr, optimizer=OPTIMIZER)
        """"""
        """Test de embeddings"""

        train_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        dev_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        test_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)


        train_data = (texts_rep_train, y_train, fnames_train)
        # dev_data = (text_test_rep, y_test, fnames_test)
        test_data = (text_test_rep, y_test, fnames_test)


        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
        iter_test = tf.data.Iterator.from_structure(test_dataset.output_types,
                                               test_dataset.output_shapes)

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_dataset)
        test_init_op = iter_test.make_initializer(test_dataset)

        epoch_start = 0
        ## Train
        sess = tf.Session()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        best_loss = 99999
        best_acc = 0
        for epoch in range(epoch_start, opts.epochs + 1):
            sess.run(train_init_op, feed_dict={
                X: train_data[0],
                y: train_data[1],
                fnames_plc: train_data[2],
                batch_size: BATCH_SIZE,
            }
                     )

            current_batch_index = 0
            next_element = iter.get_next()
            loss_count = 0
            while True:

                try:
                    data = sess.run([next_element])
                except tf.errors.OutOfRangeError:
                    break

                current_batch_index += 1
                data = data[0]
                batch_x, batch_tgt, batch_fnames = data

                _, loss_result = sess.run([train_op, loss],
                                          feed_dict={
                                              X: batch_x,
                                              y: batch_tgt,
                                              lr: train_ops.lr_scheduler(epoch),
                                              batch_size: BATCH_SIZE,
                                              is_training: True,
                                              dropout_keep_prob: 0.3,

                                          })
                # print("Loss: {}".format(loss_result))
                loss_count += loss_result

            loss_count = loss_count / current_batch_index
            logger.info("Loss on epoch {} : {} - LR: {}".format(epoch, loss_count, train_ops.lr_scheduler(epoch)))
            acc = 0
            if not by_acc:
                if loss_count < best_loss:
                    best_loss = loss_count
                    logger.info("New best_loss : {}".format(best_loss))
                    sesions.save_best(sess, opts.work_dir)
            else:

                """
                ---------------- dev
                """
                sess.run(test_init_op, feed_dict={
                    X: test_data[0],
                    y: test_data[1],
                    fnames_plc: test_data[2],
                    batch_size: BATCH_SIZE,
                }
                         )

                current_batch_index = 0
                next_element = iter_test.get_next()
                classifieds = []
                while True:

                    try:
                        data = sess.run([next_element])
                    except tf.errors.OutOfRangeError:
                        break

                    current_batch_index += 1
                    data = data[0]
                    batch_x, batch_tgt, batch_fnames = data

                    results = sess.run([softmax],
                                       feed_dict={
                                           X: batch_x,
                                           y: batch_tgt,
                                           fnames_plc: batch_fnames,
                                           batch_size: BATCH_SIZE,
                                           dropout_keep_prob: 1.0,
                                           lr: train_ops.lr_scheduler(1)
                                       })

                    acc_aux = metrics.accuracy(X=results[0], y=batch_tgt)
                    acc += acc_aux
                    for i in range(len(results[0])):
                        # to vote
                        classifieds.append(
                            (results[0][i], batch_fnames[i], batch_tgt[i])
                        )

                acc = acc / current_batch_index
                # logger.info("----------")
                # logger.info("Acc Val Test {}".format(acc))

                if acc > best_acc:
                    best_acc = acc
                    logger.info("New acc : {}".format(best_acc))
                    sesions.save_best(sess, opts.work_dir)


        """
        ----------------- TEST -----------------
        """
        logger.info("\n-- TEST --\n")
        logger.info("Restoring the best Checkpoint.")
        # restore_file = sesions.restore_from_best(sess, save_path)
        restore_file = sesions.restore_from_best(sess, opts.work_dir)
        if restore_file:
            logger.info("Best model restored")
        else:
            logger.info("Cant restore the best model.")
            exit()

        sess.run(test_init_op, feed_dict={
            X: test_data[0],
            y: test_data[1],
            fnames_plc: test_data[2],
            batch_size: BATCH_SIZE,
        }
                 )

        current_batch_index = 0
        next_element = iter_test.get_next()
        classifieds = []
        while True:

            try:
                data = sess.run([next_element])
            except tf.errors.OutOfRangeError:
                break

            current_batch_index += 1
            data = data[0]
            batch_x, batch_tgt, batch_fnames = data

            results = sess.run([softmax],
                               feed_dict={
                                   X: batch_x,
                                   y: batch_tgt,
                                   fnames_plc: batch_fnames,
                                   batch_size: BATCH_SIZE,
                                   dropout_keep_prob: 1.0,
                                   lr: train_ops.lr_scheduler(1)
                               })

            acc_aux = metrics.accuracy(X=results[0], y=batch_tgt)
            acc += acc_aux
            for i in range(len(results[0])):
                # to vote
                classifieds.append(
                    (results[0][i], batch_fnames[i], batch_tgt[i])
                )

        acc = acc / current_batch_index
        logger.info("----------")
        logger.info("Acc Val Test {}".format(acc))

        per_doc = classify_per_doc(classifieds, logger=logger)
        logger.info("----------")
        logger.info("Acc Val Test Per document votation {}".format(metrics.accuracy_per_doc(per_doc)))
        logger.info("----------")
        # [print(x) for x in classifieds_to_write]

if __name__ == "__main__":

    # Model(layers=[8,16,32], MODEL="FF", min_ngram=1, up=3) # 0.59
    # Model(layers=[8,16,32, 64], MODEL="FF", min_ngram=1, up=3)
    # Model(layers=[16, 32, 64, 128], MODEL="FF", min_ngram=1, up=3)
    # Model(layers=[16, 32, 64, 128, 256], MODEL="FF", min_ngram=1, up=3)
    Model(layers=[32, 64], MODEL="FF", min_ngram=1, up=7, max_features=50000, NUM_EPOCH=30) # 0.61
    # Model(layers=[32, 64, 128, 256,512,1024], MODEL="FF", min_ngram=1, up=7, max_features=30000, NUM_EPOCH=50) # 0.61
    # Model(filters=[32, 64, 128], MODEL="CNN", min_ngram=1, up=7, max_features=10000, NUM_EPOCH=50)
    # Model(filters=[8,16,32], MODEL="CNN", min_ngram=1, up=3, max_features=800, NUM_EPOCH=1) # 0.61
    # Model(layers=[32, 64, 128, 256, 512], MODEL="FF", min_ngram=1, up=3)
    # Model(layers=[32, 64, 128, 256, 512, 1024], MODEL="FF")
    # Model(layers=[64, 128, 256, 512, 1024], MODEL="FF")
    # Model(layers=[128, 256, 512, 1024], MODEL="FF")
    # Model(layers=[256, 512, 1024], MODEL="FF")
