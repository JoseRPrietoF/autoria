import tensorflow as tf

def save(sess, model_path):
    """
    Saves the current session to a checkpoint
    """

    saver = tf.train.Saver(max_to_keep=1)
    save_path = saver.save(sess, model_path+"model.ckpt")
    return save_path

def save_best(sess, model_path):
    """
    Saves the best session to a checkpoint
    """

    saver = tf.train.Saver(max_to_keep=1)
    save_path = saver.save(sess, model_path + "/checkpoints/best_model.ckpt")
    # save_path = saver.save(sess, model_path+"/checkpoints/best/model.ckpt")

    return save_path

def restore(sess, model_path, var_dict=None):
    """
    Restores a session from a checkpoint
    :param sess: current session instance
    :param model_path: path to file system checkpoint location
    """
    try:
        if var_dict is None:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(var_list=var_dict)
        # saver.restore(sess, model_path+"checkpoints/model.ckpt")
        saver.restore(sess, model_path+"model.ckpt")
        print("Model restored from file: %s" % model_path)
        return True
    except Exception as e:
        print("Cant restore from path {}".format(model_path))
        print(e)
        return False


def restore_from_best(sess, model_path, var_dict=None):
    """
    Restores a session from a checkpoint
    :param sess: current session instance
    :param model_path: path to file system checkpoint location
    """
    try:
        if var_dict is None:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(var_list=var_dict)
        saver.restore(sess, model_path+"/checkpoints/best_model.ckpt")
        print("Model restored from file: %s" % model_path)
        return True
    except Exception as e:
        print("Cant restore the best from path {}".format(model_path))
        return False