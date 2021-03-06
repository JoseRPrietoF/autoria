import numpy as np
import tensorflow as tf

from models import CNN
from models import CNN3D
from models import RNN
from utils import train_ops
from utils import metrics
from utils import sesions
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from canon60.functions import *

class Model:

    def __init__(self,
                 x_train, y_train, x_test, y_test,fnames_train,fnames_test,
                 x_dev, y_dev, fnames_dev,
                 embedding,
                 MAX_SEQUENCE_LENGTH,
                 EMBEDDING_DIM=300,
                 OPTIMIZER='adam',
                 logger=None,
                 opts=None,
                 by_acc=True,
                 n_classes = 2,
                 ):

        """
        Vars
        """

        MODEL = opts.model
        #FOR RNN
        HIDDEN_UNITS = 4
        NUM_LAYERS = 1

        BATCH_SIZE = 64  # Any size is accepted

        labelencoder = LabelEncoder()  # set
        y_train_ = np.array(y_train).astype(str)
        y_test_ = np.array(y_test).astype(str)
        y_dev_ = np.array(y_dev).astype(str)
        labelencoder.fit(y_train_)
        y_train_ = labelencoder.transform(y_train_)
        y_test_ = labelencoder.transform(y_test_)
        y_dev_ = labelencoder.transform(y_dev_)
        n_values = len(np.unique(y_train_))
        # To One hot
        y_train = to_categorical(y_train_, n_values)
        y_test = to_categorical(y_test_, n_values)
        y_dev = to_categorical(y_dev_, n_values)

        logger.info("texts_rep_train: {}".format(x_train.shape))
        logger.info("y_train: {}".format(x_test.shape))

        """""""""""""""""""""""""""""""""""
        # Tensorflow
        """""""""""""""""""""""""""""""""""

        batch_size = tf.placeholder(tf.int64, name="batch_size")
        if MODEL == "CNN3D":
            X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH*opts.num_tweets])
        else:
            X = tf.placeholder(tf.int64, shape=[None, MAX_SEQUENCE_LENGTH])
        fnames_plc = tf.placeholder(tf.string, shape=[None], name="fnames_plc")
        y = tf.placeholder(tf.int64, shape=[None, n_values])
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
        lr = tf.placeholder(tf.float32, shape=[])

        glove_weights_initializer = tf.constant_initializer(embedding)
        print(glove_weights_initializer)
        # embeddings = tf.Variable(
        embeddings = tf.constant(
            # tf.random_uniform([2000, EMBEDDING_DIM], -1.0, 1.0)
            embedding
        )

        print(embeddings)
        """
        GET THE MODEL
        """
        if MODEL == "CNN":
            logits = CNN.get_model(X, W=embeddings, is_training=is_training, filters=opts.filters, n_classes=n_classes, logger=logger)
        elif MODEL == "RNN":
            logits = RNN.get_model(X, W=embeddings,dropout_keep_prob=dropout_keep_prob, hidden_size = HIDDEN_UNITS, n_classes=n_classes, num_layers=NUM_LAYERS)
        elif MODEL == "CNN3D":
            x_train = x_train.reshape((-1, x_train.shape[-1] * opts.num_tweets))
            x_dev = x_dev.reshape((-1, x_dev.shape[-1] * opts.num_tweets))
            x_test = x_test.reshape((-1, x_test.shape[-1] * opts.num_tweets))

            logits = CNN3D.get_model(X,
                                     W=embeddings,
                                     is_training=is_training, filters=opts.filters,
                                     n_classes=n_classes, logger=logger
              )
        print(logits)
        softmax = tf.nn.softmax(logits)

        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        print("{} params to train".format(num_params))

        train_op, loss = train_ops.train_op(logits,y=y, learning_rate=lr, optimizer=OPTIMIZER)
        # exit()
        """"""
        """Test de embeddings"""

        train_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        dev_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        test_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        print(x_train.shape)
        print(y_train.shape)
        print(len(fnames_train))
        train_data = (x_train, y_train, fnames_train)
        dev_data = (x_dev, y_dev, fnames_dev)
        test_data = (x_test, y_test, fnames_test)


        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
        iter_dev = tf.data.Iterator.from_structure(dev_dataset.output_types,
                                                   dev_dataset.output_shapes)
        iter_test = tf.data.Iterator.from_structure(test_dataset.output_types,
                                                    test_dataset.output_shapes)

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_dataset)
        dev_init_op = iter_dev.make_initializer(dev_dataset)
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
                sess.run(dev_init_op, feed_dict={
                    X: dev_data[0],
                    y: dev_data[1],
                    fnames_plc: dev_data[2],
                    batch_size: BATCH_SIZE,
                }
                         )

                current_batch_index = 0
                next_element = iter_dev.get_next()
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

                acc = acc / current_batch_index
                logger.info("----------")
                logger.info("Loss on epoch {} : {} - LR: {}".format(epoch, loss_count, train_ops.lr_scheduler(epoch)))
                logger.info("Acc Val Test {}".format(acc))

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
            results = results[0]

            acc_aux = metrics.accuracy(X=results, y=batch_tgt)
            acc += acc_aux
            for i in range(len(results)):
                hyp = [np.argmax(results[i], axis=-1)]
                hyp = labelencoder.inverse_transform(hyp)[0]  # real label   #set

                doc_name = batch_fnames[i].decode("utf-8").split("/")[-1]

                tgt = [np.argmax(batch_tgt[i], axis=-1)]
                tgt = labelencoder.inverse_transform(tgt)[0]  # real label   #set

                # to vote
                classifieds.append(
                    (hyp, doc_name, tgt)
                )
            # print(classifieds)
            # exit()

        acc = acc / current_batch_index
        logger.info("----------")
        logger.info("Acc Val Test {}".format(acc))
        #
        # per_doc = classify_per_doc(classifieds, logger=logger)
        # logger.info("----------")
        # logger.info("Acc Val Test Per document votation {}".format(metrics.accuracy_per_doc(per_doc)))
        # logger.info("----------")
        # [print(x) for x in classifieds_to_write]
        self.results = classifieds

    def get_results(self):
        pred, true = [], []
        for result in self.results:
            pred.append(result[0])
            true.append(result[2])
        return pred, true