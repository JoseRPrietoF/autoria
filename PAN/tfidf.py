import numpy as np
import tensorflow as tf
from data import process
from models import CNN
from models import FF
from models import RNN
from utils import train_ops
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from utils import metrics
from utils.sesions import *

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
                 NUM_EPOCH=50,
                 min_ngram=1, up=5,max_features=None,
                 dataset="PAN2019",
                 logger=None,
                 opts=None,
                 DEBUG=False, lang="es"
                 ):

        """
        Vars
        """
        BATCH_SIZE=opts.batch_size
        logger = logger or logging.getLogger(__name__)
        # MODEL = "RNN"
        # MODEL = "CNN"
        if not DEBUG:

            ## PAN
            path = opts.tr_data+'/'+lang
            path_test = opts.i+'/'+lang
            if do_val:
                txt_train = opts.file_i+"/{}/truth-train.txt".format(lang)
                txt_dev = opts.file_i+"/{}/truth-dev.txt".format(lang)
                dt_train = process.PAN2019(path=path, txt=txt_train, join_all= MODEL == "FF")
                dt_dev = process.PAN2019(path=path, txt=txt_dev, join_all= MODEL == "FF")
                fnames_dev = dt_dev.fnames
                y_dev = dt_dev.y
                x_dev = dt_dev.X
                del dt_dev
            else:
               txt_train = opts.file_i+"/{}/truth.txt".format(lang)
               dt_train = process.PAN2019(path=path, txt=txt_train, join_all= MODEL == "FF")

            dt_test = process.PAN2019_Test(path=path_test, join_all= MODEL == "FF")
            n_classes = 2 # bot or not bot


            x_train = dt_train.X
            y_train = dt_train.y
            print(len(x_train))
            print(len(y_train))

            x_test = dt_test.X
            # y_test = dt_test.y

            fnames_train = dt_train.fnames
            fnames_test = dt_test.fnames

            labelencoder = LabelEncoder()  #set
            y_train_ = np.array(y_train).astype(str) 
            # y_test_ = np.array(y_test).astype(str) 
            labelencoder.fit(y_train_)
            y_train_ = labelencoder.transform(y_train_)
            # y_test_ = labelencoder.transform(y_test_)
            n_values = len(np.unique(y_train_))
            # To One hot
            y_train = to_categorical(y_train_, n_values)


            # y_test = to_categorical(y_test_, n_values)


            if max_features:

                rep = TfidfVectorizer(ngram_range=(min_ngram,up),max_features=max_features)
            else:
                rep = TfidfVectorizer(ngram_range=(min_ngram,up))

            del dt_train
            del dt_test

            logger.info("fit_transform tfidf")
            texts_rep_train = rep.fit_transform(x_train)
            logger.info("To array")
            texts_rep_train = texts_rep_train.toarray()

            logger.info("transform tfidf")
            text_test_rep = rep.transform(x_test)
            logger.info("To array")
            text_test_rep = text_test_rep.toarray()
            if do_val:
                text_dev_rep = rep.transform(x_dev)
                text_dev_rep = text_dev_rep.toarray()
                y_dev_ = np.array(y_dev).astype(str)
                y_dev_ = labelencoder.transform(y_dev_)
                y_dev = to_categorical(y_dev_, n_values)


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
        test_dataset = tf.data.Dataset.from_tensor_slices((X, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)


        train_data = (texts_rep_train, y_train, fnames_train)
        if do_val:
            dev_data = (text_dev_rep, y_dev, fnames_dev)
        test_data = (text_test_rep, fnames_test)
        print(text_test_rep.shape)
        print(len(fnames_test))

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
        iter_test = tf.data.Iterator.from_structure(test_dataset.output_types,
                                               test_dataset.output_shapes)

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_dataset)
        dev_init_op = iter.make_initializer(dev_dataset)
        test_init_op = iter_test.make_initializer(test_dataset)

        epoch_start = 0
        ## Train
        sess = tf.Session()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        best_acc = 0
        for epoch in range(epoch_start, NUM_EPOCH + 1):
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
            if do_val:
                print("Eval")
                ## Eval
                sess.run(dev_init_op, feed_dict={
                # sess.run(dev_init_op, feed_dict={
                    X: dev_data[0],
                    y: dev_data[1],
                    fnames_plc: dev_data[2],
                    batch_size: BATCH_SIZE,
                }
                         )

                current_batch_index = 0
                next_element = iter.get_next()


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
                                                  lr: train_ops.lr_scheduler(epoch),
                                                  batch_size: BATCH_SIZE,
                                                  is_training: False,
                                                  dropout_keep_prob: 1.0
                                              })
                    results = results[0]
                    acc_aux = metrics.accuracy(X=results, y=batch_tgt)
                    acc += acc_aux

                acc = acc / current_batch_index
                print("Acc Val epoch {} : {}".format(epoch, acc))
                print("----------")
                if acc > best_acc:
                    best_acc = acc
                    logger.info("New acc : {}".format(best_acc))
                    save_best(sess, opts.work_dir)
        if opts.testing and opts.do_val:
            logger.info("Model: {}".format(MODEL))
            logger.info("layers: {}".format(layers))
            logger.info("max_features: {}".format(max_features))
            logger.info("Min and max features: {} - {}".format(min_ngram, up))
            logger.info("Best acc : {}".format(best_acc))
            exit()
        """
        ----------------- TEST -----------------
        """
        logger.info("\n-- TEST --\n")
        logger.info("Restoring the best Checkpoint.")
        # restore_file = sesions.restore_from_best(sess, save_path)
        restore_file = restore_from_best(sess, opts.work_dir)
        if restore_file:
            logger.info("Best model restored")
        else:
            logger.info("Cant restore the best model.")
            exit()
        Y_FALSA = np.random.randint(1, size=(BATCH_SIZE, n_classes))
        print(Y_FALSA.shape)
        logger.info("\n-- TEST --\n")

        sess.run(test_init_op, feed_dict={
            X: test_data[0],
            y: Y_FALSA,
            fnames_plc: test_data[1],
            batch_size: BATCH_SIZE,
        }
                 )

        current_batch_index = 0
        next_element = iter_test.get_next()
        loss_count = 0
        classifieds = []
        classifieds_to_write = []
        while True:

            try:
                data = sess.run([next_element])
            except tf.errors.OutOfRangeError:
                break

            current_batch_index += 1
            data = data[0]
            batch_x, batch_fnames = data

            results = sess.run([softmax],
                               feed_dict={
                                   X: batch_x,
                                   y: Y_FALSA,
                                   batch_size: BATCH_SIZE,
                                   dropout_keep_prob: 1.0,
                                   lr: train_ops.lr_scheduler(1)
                               })



            for i in range(len(results[0])):
                # to write
                hyp = [np.argmax(results[0][i], axis=-1)]
                hyp = labelencoder.inverse_transform(hyp)[0] #real label   #set
                doc_name = batch_fnames[i].decode("utf-8").split("/")[-1]

                classifieds_to_write.append((
                    doc_name, lang, hyp
                ))

        logger.info("----------")
        logger.info("Writting results in output dir {}".format("{}/{}".format(opts.o, lang)))
        process.write_from_array(classifieds_to_write, "{}/{}".format(opts.o, lang))
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
